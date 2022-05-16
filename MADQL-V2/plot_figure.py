import numpy as np
import matplotlib.pyplot as plt

from utils import loadData, cdf


def plotFinalReportCapacity():
    MADQL = loadData(path="./simulation_data/reward-data-001.txt", name="MADQL-average-rewards")
    random = loadData(path="./simulation_data/reward-data-001.txt", name="Random-average-rewards")
    cellES = loadData(path="./simulation_data/reward-data-001.txt", name="CELL_ES-average-rewards")
    maxPower = loadData(path="./simulation_data/reward-data-001.txt", name="MaxPower-average-rewards")

    cdf(MADQL, color="red", linestyle=":", label="MADQL")
    cdf(random, color="blue", linestyle="--", label="Random")
    cdf(cellES, color="green", linestyle="-.", label="Cell ES")
    cdf(maxPower, color="orange", linestyle="-", label="Max Power")

    averageY = np.zeros(10) + 0.5
    averageX = np.linspace(0, 3.8, 10)
    plt.plot(averageX, averageY, 'k--')

    verticalY = np.linspace(-0.05, 0.5, 10)
    maxPowerX = np.zeros(10) + sum(maxPower) / len(maxPower)
    MADQLX = np.zeros(10) + 3.69
    cellESX = np.zeros(10) + 3.8
    plt.plot(maxPowerX, verticalY, 'k--')
    plt.plot(MADQLX, verticalY, 'k--')
    plt.plot(cellESX, verticalY, 'k--')

    plt.xlim([0.3, 6.5])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("System Capacity (bps/Hz)")
    plt.ylabel("CDF of Reward")
    plt.legend(loc='upper left')
    plt.show()
