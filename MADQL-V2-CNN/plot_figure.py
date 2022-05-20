import numpy as np
import matplotlib.pyplot as plt

from utils import loadData, cdf, pdf
from channel import Channel


def plotFinalReportCapacity():
    DATA_PATH = "./simulation_data/data.txt"

    MADQL = loadData(path=DATA_PATH, name="MADQL-average-rewards")
    random = loadData(path=DATA_PATH, name="Random-average-rewards")
    cellES = loadData(path=DATA_PATH, name="CELL_ES-average-rewards")
    maxPower = loadData(path=DATA_PATH, name="MaxPower-average-rewards")

    cdf(MADQL, color="red", linestyle=":", label="MADQL")
    cdf(random, color="blue", linestyle="--", label="Random")
    cdf(cellES, color="green", linestyle="-.", label="Cell ES")
    cdf(maxPower, color="orange", linestyle="-", label="Max Power")

    plt.xlim([0.3, 6.5])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("System Capacity (bps/Hz)")
    plt.ylabel("CDF of Reward")
    plt.legend(loc='upper left')
    plt.show()


def plotRicianChannel():
    """[test] pdf of channel"""
    channel = Channel([0., 0., 10.], [100., 100., 1.5])

    channel.setRicianFactor(10)
    channels = []
    for i in range(5000):
        norm = np.linalg.norm(channel.getCSI())
        channels.append(norm)
        channel.update()
    pdf(channels, linestyle=":", label="K=10")

    channel.setRicianFactor(5)
    channels = []
    for i in range(5000):
        norm = np.linalg.norm(channel.getCSI())
        channels.append(norm)
        channel.update()
    pdf(channels, linestyle="-", label="K=5")

    channel.setRicianFactor(2)
    channels = []
    for i in range(5000):
        norm = np.linalg.norm(channel.getCSI())
        channels.append(norm)
        channel.update()
    pdf(channels, linestyle="--", label="K=2")

    plt.legend(loc='upper right')
    plt.xlabel("Norm of Channel")
    plt.ylabel("Probability")
    plt.show()
