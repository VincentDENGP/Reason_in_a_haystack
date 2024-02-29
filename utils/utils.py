import random
import numpy as np
import json
import yaml
from src.config import LLMNeedleHaystackTesterArgs

def save_config_as_json(config):
    # 将配置对象转换为字典
    config_dict = vars(config)

    # 遍历字典，将NumPy数组转换为列表
    for key, value in config_dict.items():
        if isinstance(value, np.ndarray):
            config_dict[key] = value.tolist()

    # 创建JSON文件的路径
    json_file_path = f"{config.save_prefix}/config.json"

    # 将配置字典保存为JSON文件
    with open(json_file_path, 'w') as json_file:
        json.dump(config_dict, json_file, ensure_ascii=False, indent=4)

def generate_random_number(num_digits):
    lower_bound = 10**(num_digits - 1)
    upper_bound = 10**num_digits - 1
    return random.randint(lower_bound, upper_bound)

def logistic(x, L=100, x0=50, k=.1):
    if x == 0:
        return 0
    if x == 100:
        return 100
    return np.round(L / (1 + np.exp(-k * (x - x0))), 3)


def load_config(config_path):
    """从YAML文件加载配置。"""
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    args_instance = LLMNeedleHaystackTesterArgs(**config)
    return args_instance