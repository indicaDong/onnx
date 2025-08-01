import os
import numpy as np
import onnxruntime as ort
from perfectdou.env.encode import (
    encode_obs_landlord,
    encode_obs_peasant,
    _decode_action,
)
from perfectdou.env.game import bombs


def _load_model(position):
    model_dir = "{}/../model/onnx_v7_fixed".format(os.path.dirname(__file__))
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.inter_op_num_threads = 1
    sess_options.intra_op_num_threads = 1
    sess_options.log_severity_level = 3
    return ort.InferenceSession("{}/{}_v7.onnx".format(model_dir, position), sess_options)


RLCard2EnvCard = {
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
    "2": 17,
    "B": 20,
    "R": 30,
}


class onnxdouAgentV5:
    def __init__(self, position):
        self.model = _load_model(position)
        self.position = position
        self.bomb_num = 0
        self.control = 0
        self.have_bomb = 0
        self.num_tta_runs = 10
        self.use_noise = True
        self.use_dropout = True
        self.noise_scale = 0.02
        self.dropout_rate = 0.4
        

        
        # 检查模型的输出名称
        self.output_name = self.model.get_outputs()[0].name


    def act(self, infoset):
        if infoset.player_position == "landlord":
            obs = encode_obs_landlord(infoset)
        elif infoset.player_position == "landlord_up":
            obs = encode_obs_peasant(infoset)
        elif infoset.player_position == "landlord_down":
            obs = encode_obs_peasant(infoset)
        input_name = self.model.get_inputs()[0].name
        input_data = np.concatenate(
            [obs["x_no_action"].flatten(), obs["legal_actions_arr"].flatten()]
        ).reshape(1, -1).astype(np.float32)

        use_randomization = self.use_noise or self.use_dropout
        all_results = []
        for i in range(self.num_tta_runs):

            current_input = input_data
            logit = self.model.run([self.output_name], {input_name: current_input})

            all_results.append(logit[0])
        averaged_result = np.mean(all_results, axis=0)
        action_id = np.argmax(averaged_result)


        action = _decode_action(action_id, obs["current_hand"], obs["actions"])
        action = [] if action == "pass" else [RLCard2EnvCard[e] for e in action]
        print("action:", action)

        return self.find_sublist_index(infoset.legal_actions, action)

    def act_with_details(self, infoset):
        """返回包含详细logit信息的字典"""
        if infoset.player_position == "landlord":
            obs = encode_obs_landlord(infoset)
        elif infoset.player_position == "landlord_up":
            obs = encode_obs_peasant(infoset)
        elif infoset.player_position == "landlord_down":
            obs = encode_obs_peasant(infoset)
        input_name = self.model.get_inputs()[0].name
        input_data = np.concatenate(
            [obs["x_no_action"].flatten(), obs["legal_actions_arr"].flatten()]
        ).reshape(1, -1).astype(np.float32)

        # 检查输入数据范围
        print(f"输入数据范围: min={np.min(input_data)}, max={np.max(input_data)}, mean={np.mean(input_data):.4f}")
        print(f"输入数据包含inf: {np.any(np.isinf(input_data))}, 包含nan: {np.any(np.isnan(input_data))}")

        use_randomization = self.use_noise or self.use_dropout
        all_results = []
        print(f"TTA运行信息:")
        print(f"  运行次数: {self.num_tta_runs}")
        for i in range(self.num_tta_runs):

            current_input = input_data
   
            logit = self.model.run([self.output_name], {input_name: current_input})
            print("logit:", logit[0])
            all_results.append(logit[0])
        
        averaged_result = np.mean(all_results, axis=0)
        action_id = np.argmax(averaged_result)
        
        action = _decode_action(action_id, obs["current_hand"], obs["actions"])
        action = [] if action == "pass" else [RLCard2EnvCard[e] for e in action]
        
        action_index = self.find_sublist_index(infoset.legal_actions, action)
        
        # 显示所有合法动作
        print(f"所有合法动作:")
        for i, legal_action in enumerate(infoset.legal_actions):
            marker = " ← 选择" if i == action_index else ""
            print(f"  [{i}]: {legal_action}{marker}")
        
        return {
            'action_index': action_index,
            'action': action,
            'action_id': int(action_id),
            'logit': averaged_result.flatten(),
            'max_logit_value': float(np.max(averaged_result)),
            'all_results': all_results  # 包含所有TTA运行的结果
        }


    def find_sublist_index(self, nested_list, target_sublist):
        for idx1, sublist in enumerate(nested_list):
            if len(target_sublist) == len(sublist):
                same = True
                for idx2, value in enumerate(sublist):
                    if value != target_sublist[idx2]:
                        same = False
                        break
                if same:
                    return idx1
        return -1  # 未找到时返回-1
