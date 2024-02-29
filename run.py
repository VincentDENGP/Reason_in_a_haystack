from src.tester import LLMNeedleHaystackTester
from src.config import LLMNeedleHaystackTesterArgs
from utils.vis import result_analysis
from utils.utils import save_config_as_json, load_config
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='加载YAML配置文件的脚本')
    parser.add_argument('--config', type=str, help='配置文件的路径')
    args = parser.parse_args()
    config = load_config(args.config)
    tester = LLMNeedleHaystackTester(config)
    tester.start_test()
    save_config_as_json(config)
    if config.pure_cal:
        pass#TODO, add acc analysis
    else:
        result_analysis(config.save_prefix+'/results', config.model_name, config.inject, config.mode, config.save_prefix+'/scores3.png', figure_suffix= "128K Context\nFact Retrieval Across Context Lengths ('Needle In A HayStack')", wchich_parser = "score3", context_length = 32000)