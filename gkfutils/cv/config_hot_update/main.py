import time
from hot_config import HotConfig

if __name__ == "__main__":
    # 全局单例配置（整个程序共用一个）
    config = HotConfig("gkfutils/cv/config_hot_update/config.yaml")
    print("程序运行中，可随时修改 config.yaml 并保存...")

    while True:
        # 这里每次读取都是最新的！
        model_name = config["model"]["name"]
        conf = config["model"]["conf_thres"]
        port = config["server"]["port"]

        print(f"当前配置：model={model_name}, conf={conf}, port={port}")
        time.sleep(1)