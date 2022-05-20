from channel import Channel
from utils import generateChannelIndex


def generateChannel(sectors, UEs):
    """
    Channel Generator
    Args:
        sectors: list of sector
        UEs: list of UE
    Returns:
        channels, dictionary of channel, "SectorIndex-UEIndex" -> Channel
    """
    channels = {}
    for sector in sectors:
        for UE in UEs:
            channelIndex = generateChannelIndex(sector.getIndex(), UE.getIndex())
            channels[channelIndex] = Channel(sector.getPosition(), UE.getPosition())
    return channels
