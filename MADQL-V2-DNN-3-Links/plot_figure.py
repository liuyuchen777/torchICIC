import numpy as np
import matplotlib.pyplot as plt

from utils import loadData, cdf, pdf, average, mid
from channel import Channel


def plotTempReportCapacity():
    DATA_PATH = "simulation_data/reward-data-003.txt"

    MADQL = loadData(path=DATA_PATH, name="Algorithm.MADQL-averageCapacity")
    MADQL = MADQL[-2000:]
    random = loadData(path=DATA_PATH, name="Algorithm.RANDOM-averageCapacity")
    maxPower = loadData(path=DATA_PATH, name="Algorithm.MAX_POWER-averageCapacity")
    cellES = loadData(path=DATA_PATH, name="Algorithm.CELL_ES-averageCapacity")

    cdf(MADQL, color="red", linestyle=":", label="MADQL")
    cdf(random, color="blue", linestyle="--", label="Random")
    cdf(maxPower, color="orange", linestyle="-", label="Max Power")
    cdf(cellES, color="green", linestyle="-.", label="Cell ES")

    plt.xlabel("System Capacity (bps/Hz)")
    plt.ylabel("CDF of Reward")
    plt.legend(loc='upper left')
    plt.show()


def plotFinalReportCapacity():
    DATA_PATH = "./simulation_data/reward-data-001.txt"

    MADQL = loadData(path=DATA_PATH, name="MADQL-average-rewards")
    random = loadData(path=DATA_PATH, name="Random-average-rewards")
    cellES = loadData(path=DATA_PATH, name="CELL_ES-average-rewards")
    maxPower = loadData(path=DATA_PATH, name="MaxPower-average-rewards")

    cdf(MADQL, color="red", linestyle=":", label="MADQL")
    cdf(random, color="blue", linestyle="--", label="Random")
    cdf(cellES, color="green", linestyle="-.", label="Cell ES")
    cdf(maxPower, color="orange", linestyle="-", label="Max Power")

    averageY = np.zeros(10) + 0.5
    averageX = np.linspace(0, mid(cellES), 10)
    plt.plot(averageX, averageY, 'k--')

    verticalY = np.linspace(-0.05, 0.5, 10)
    maxPowerX = np.zeros(10) + mid(maxPower)
    MADQLX = np.zeros(10) + mid(MADQL)
    cellESX = np.zeros(10) + mid(cellES)
    plt.plot(maxPowerX, verticalY, 'k--')
    plt.plot(MADQLX, verticalY, 'k--')
    plt.plot(cellESX, verticalY, 'k--')

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


def plotRewardChange():
    MADQL = loadData(path="simulation_data/reward-data-003.txt", name="Algorithm.MADQL-averageCapacity")
    averageMADQL = []
    PAGE = 100
    temp = []
    for data in MADQL:
        if len(temp) < PAGE:
            temp.append(data)
        else:
            averageMADQL.append(sum(temp) / len(temp))
            temp = []
    timeSlot = range(495)

    randomAverage = np.zeros(495) + 3.97
    cellESAverage = np.zeros(495) + 7.26

    plt.plot(timeSlot, randomAverage, linestyle=":")
    plt.plot(timeSlot, cellESAverage, linestyle="--")
    plt.plot(timeSlot, averageMADQL, color="Blue")

    plt.title("System Capacity Average via 100 Time Slot")
    plt.ylabel("System Capacity (bps/Hz)")
    plt.xlabel("Time Slot")
    plt.show()


def calIndicator(dataPath):
    MADQL = loadData(path=dataPath, name="Algorithm.MADQL-averageCapacity")
    MADQL = MADQL[-2000:]
    random = loadData(path=dataPath, name="Algorithm.RANDOM-averageCapacity")
    maxPower = loadData(path=dataPath, name="Algorithm.MAX_POWER-averageCapacity")
    cellES = loadData(path=dataPath, name="Algorithm.CELL_ES-averageCapacity")

    print(f"MADQL average: {average(MADQL)}, mid: {mid(MADQL)}")
    print(f"Random average: {average(random)}, mid: {mid(random)}")
    print(f"Max Power average: {average(maxPower)}, mid: {mid(maxPower)}")
    print(f"Cell ES average: {average(cellES)}, mid: {mid(cellES)}")
