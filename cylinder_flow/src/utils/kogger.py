import os
import logging
import sys


class Logger:
    """复刻 kogger 的 Logger 类，并兼容 file 参数"""

    def __init__(self, name='root', file=None):  # 这里把 stream 改为了 file
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.name = name

        # 清除原有 handler
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # 定义格式
        formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')

        # 如果指定了 file，则输出到该文件路径
        if file:
            # --- 关键修改：自动创建父目录 ---
            log_dir = os.path.dirname(file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            # ------------------------------
            handler = logging.FileHandler(file)
        else:
            handler = logging.StreamHandler(sys.stdout)

        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def info(self, msg): self.logger.info(
        msg.upper() if isinstance(msg, str) else msg)

    def warning(self, msg): self.logger.warning(
        msg.upper() if isinstance(msg, str) else msg)
    def error(self, msg): self.logger.error(
        msg.upper() if isinstance(msg, str) else msg)


# --- 模拟 kogger 的全局单例行为 ---
_global_logger = Logger('root')


def set_name(name):
    global _global_logger
    _global_logger = Logger(name)


def set_file(filename):
    global _global_logger
    _global_logger = Logger(_global_logger.logger.name, stream=filename)


def info(msg): _global_logger.info(msg)
def warning(msg): _global_logger.warning(msg)
def error(msg): _global_logger.error(msg)
