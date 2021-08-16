from config import Config
from utils import Logger


class Environment:
    def __init__(self, CUs, config=Config, logger=Logger()):
        print("Under Construct")
        self.CUs = CUs
        self.channels = []
        self.config = config
        self.logger = logger

    def _init_channel_(self):
        self.logger.log_d("--------------start to init channels---------------")

    def get_channel(self):
        return self.channels

    def print_channel_info(self):
        print("Under Construct")


if __name__ == "__main__":
    logger = Logger(debug_tag=True)
    env = Environment(logger=logger)


