import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from utils import loadData, cdf, pdf, average, mid


"""Element Plot"""


def plotCapacityCDF(dataPath, dataName, dataNumber, labelName, lineStyle):
    data = loadData(dataPath, dataName)
    data = data[-1 * dataNumber:]
    cdf(data, lineStyle=lineStyle, label=labelName)


def calAndPrintIndicator(dataPath, dataName, dataNumber):
    data = loadData(path=dataPath, name=dataName)
    data = data[-1 * dataNumber:]
    print(f"{dataName} average: {average(data)}, mid: {mid(data)}")


def plotRewardChange(dataPath, dataName, lineStyle):
    data = loadData(path=dataPath, name=dataName)

    data = windowAverage(data, N=500)
    timeSlot = range(len(data))

    plt.plot(timeSlot, data, lineStyle=lineStyle)


def savgolAverage(data, windowLen, polyOrder):
    return savgol_filter(data, window_length=windowLen, polyorder=polyOrder)


def windowAverage(data, N):
    """average on previous N time slot"""
    averageData = []
    for i in range(len(data)):
        sumStart = i-N if (i-N) > 0 else 0
        sumLen = N if (i-N) > 0 else i + 1
        averageData.append(sum(data[sumStart:i+1])/sumLen)
    return averageData


"""Aggregation Plot"""


def plotDifferentAlpha(dataPath):
    plotCapacityCDF(dataPath, dataName="alpha0.01-Algorithm.MADQL-averageCapacity", dataNumber=1000,
                    labelName="Alpha=0.01", lineStyle="-")
    plotCapacityCDF(dataPath, dataName="alpha0.1-Algorithm.MADQL-averageCapacity", dataNumber=1000,
                    labelName="Alpha=0.1", lineStyle="-")
    plotCapacityCDF(dataPath, dataName="alpha0.5-Algorithm.MADQL-averageCapacity", dataNumber=1000,
                    labelName="Alpha=0.5", lineStyle="--")
    plotCapacityCDF(dataPath, dataName="alpha10-Algorithm.MADQL-averageCapacity", dataNumber=1000,
                    labelName="Alpha=1", lineStyle=":")
    plotCapacityCDF(dataPath, dataName="alpha2-Algorithm.MADQL-averageCapacity", dataNumber=1000,
                    labelName="Alpha=2", lineStyle=":")
    plotCapacityCDF(dataPath, dataName="alpha1-Algorithm.MADQL-averageCapacity", dataNumber=1000,
                    labelName="Alpha=10", lineStyle="-.")

    # calAndPrintIndicator(dataPath, dataName="alpha0.1-Algorithm.MADQL-averageCapacity", dataNumber=500)
    # calAndPrintIndicator(dataPath, dataName="alpha0.5-Algorithm.MADQL-averageCapacity", dataNumber=500)
    # calAndPrintIndicator(dataPath, dataName="alpha10-Algorithm.MADQL-averageCapacity", dataNumber=500)

    plt.xlabel("System Capacity (bps/Hz)")
    plt.ylabel("CDF of Reward")
    plt.legend(loc='upper left')
    plt.title("Average System Capacity CDF with Different Interference Penalty")
    plt.show()


def plotTempReportCapacity(dataPath):
    plotCapacityCDF(dataPath, dataName="Algorithm.MADQL-averageCapacity", dataNumber=2000,
                    labelName="MADQL", lineStyle="-")
    plotCapacityCDF(dataPath, dataName="Algorithm.RANDOM-averageCapacity", dataNumber=2000,
                    labelName="Random", lineStyle="--")
    plotCapacityCDF(dataPath, dataName="Algorithm.MAX_POWER-averageCapacity", dataNumber=2000,
                    labelName="Max Power", lineStyle="-.")
    plotCapacityCDF(dataPath, dataName="Algorithm.CELL_ES-averageCapacity", dataNumber=2000,
                    labelName="Cell ES", lineStyle=":")

    plt.xlabel("System Capacity (bps/Hz)")
    plt.ylabel("CDF of Reward")
    plt.legend(loc='upper left')
    plt.show()


def plotMADQLRewardChange(dataPath):
    # plotRewardChange(dataPath, dataName="alpha0.1-Algorithm.MADQL-averageCapacity", lineStyle="-")
    # plotRewardChange(dataPath, dataName="alpha0.5-Algorithm.MADQL-averageCapacity", lineStyle="--")
    plotRewardChange(dataPath, dataName="Algorithm.MADQL-averageCapacity", lineStyle="-")

    plt.title("System Capacity Average via Time Slot")
    plt.ylabel("System Capacity (bps/Hz)")
    plt.xlabel("Time Slot")
    plt.show()


if __name__ == "__main__":
    plotDifferentAlpha("./simulation_data/data.txt")
    # plotMADQLRewardChange("./simulation_data/reward-data-003.txt")
    # plotMADQLRewardChange("./simulation_data/data.txt")
