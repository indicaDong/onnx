# 下载 Miniconda（Python 3.7 专用版本）
!wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh

# 静默安装（无需交互）
!bash Miniconda3-py37_4.10.3-Linux-x86_64.sh -b -f -p /usr/local/miniconda

# 将 conda 添加到系统 PATH
import os
os.environ['PATH'] = "/usr/local/miniconda/bin:" + os.environ['PATH']

# 验证安装
!conda --version

# 创建名为 `py37` 的独立环境
!conda create -n py37 python=3.7 -y

# 激活环境
!source /usr/local/miniconda/bin/activate py37

# 验证 Python 版本
!python --version