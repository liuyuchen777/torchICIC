import logging
from config import Config
from utils import index2str
from channel import Channel


class Environment:
    def __init__(self, CUs):
        self.logger = logging.getLogger(__name__)
        self.config = Config()
        self.CUs = CUs
        self.channels = dict()  # dict(channel_index) -> Channel
        self._init_channel_()

    def _init_channel_(self):
        """
        build all channels
        """
        for cu in self.CUs:
            for sector in cu.sectors:
                # intra CU
                for ue in cu.UEs:
                    channel = Channel(sector, ue)
                    self.channels[index2str(channel.index)] = channel
                # inter CU
                for other_cu in self.CUs:
                    if other_cu.index == cu.index:
                        continue
                    for ue in other_cu.UEs:
                        channel = Channel(sector, ue)
                        self.channels[index2str(channel.index)] = channel

    def get_channel(self):
        return self.channels

    def print_channel(self):
        for (k, v) in self.channels.items():
            print(f'CSI of index {k}:')
            print(f'{v.get_csi()}')

    def cal_reward(self):
        """
        use action and CSI calculate each CU's reward
        return in list, length of CU number
        """
        reward = []
        for cu in self.CUs:
            print("Under Construct")
        return reward



