import yaml
import time
import os
from threading import Thread


class HotConfig:
    def __init__(self, config_path="config.yaml", check_interval=1):
        self.config_path = config_path
        self.check_interval = check_interval  # 检查间隔（秒）
        self.last_mtime = 0  # 上次文件修改时间
        self.config = {}  # 配置内容
        self.load_config()  # 首次加载
        self.start_watch_thread()  # 启动监听线程

    def load_config(self):
        """加载 YAML 配置"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
            self.last_mtime = os.path.getmtime(self.config_path)
            print(f"✅ 配置已加载: {self.config_path}")
        except Exception as e:
            print(f"❌ 加载配置失败: {e}")

    def check_update(self):
        """检查文件是否被修改"""
        current_mtime = os.path.getmtime(self.config_path)
        if current_mtime != self.last_mtime:
            self.load_config()

    def watch_loop(self):
        """无限循环监听文件变化"""
        while True:
            self.check_update()
            time.sleep(self.check_interval)

    def start_watch_thread(self):
        """启动后台监听线程"""
        thread = Thread(target=self.watch_loop, daemon=True)
        thread.start()
        print(f"🔍 配置热更新已启动，监听文件：{self.config_path}")

    # 让你可以像 dict 一样使用 config
    def __getitem__(self, key):
        return self.config.get(key)

    def get(self, key, default=None):
        return self.config.get(key, default)

