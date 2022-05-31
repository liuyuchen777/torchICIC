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

    data = savgolAverage(data, windowLen=701, polyOrder=5)
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


def plotCapacityCDFCompare(dataPath):
    plotCapacityCDF(dataPath, dataName="3-Links-Test-Algorithm.MADQL-averageCapacity", dataNumber=2000,
                    labelName="MADQL", lineStyle="-")
    plotCapacityCDF("./simulation_data/reward-data-009.txt", dataName="default-Algorithm.RANDOM-averageCapacity", dataNumber=2000,
                    labelName="Random", lineStyle="--")
    plotCapacityCDF("./simulation_data/reward-data-009.txt", dataName="power-Algorithm.CELL_ES-averageCapacity", dataNumber=2000,
                    labelName="Power ES", lineStyle="-.")
    plotCapacityCDF("./simulation_data/reward-data-009.txt", dataName="beam-Algorithm.CELL_ES-averageCapacity", dataNumber=2000,
                    labelName="Beam ES", lineStyle=":")
    plotCapacityCDF("./simulation_data/reward-data-009.txt", dataName="default-Algorithm.CELL_ES-averageCapacity", dataNumber=2000,
                    labelName="Power and Beam ES", lineStyle=":")

    calAndPrintIndicator(dataPath, dataName="3-Links-Test-Algorithm.MADQL-averageCapacity", dataNumber=2000)
    calAndPrintIndicator("./simulation_data/reward-data-009.txt", dataName="default-Algorithm.CELL_ES-averageCapacity", dataNumber=2000)

    plt.xlabel("System Capacity (bps/Hz)")
    plt.ylabel("CDF of Reward")
    plt.legend(loc='lower right')
    plt.show()


def plot21LinkCDFCompare():
    plotCapacityCDF("./simulation_data/reward-data-013.txt", dataName="default-Algorithm.MADQL-averageCapacity",
                    dataNumber=2000, labelName="MADQL", lineStyle="-")
    plotCapacityCDF("./simulation_data/reward-data-013.txt", dataName="default-Algorithm.RANDOM-averageCapacity",
                    dataNumber=2000, labelName="Random", lineStyle="--")
    plotCapacityCDF("./simulation_data/reward-data-017.txt", dataName="default-Algorithm.CELL_ES-averageCapacity",
                    dataNumber=2000, labelName="Local ES", lineStyle="-.")

    plt.xlabel("System Capacity (bps/Hz)")
    plt.ylabel("CDF of Reward")
    plt.legend(loc='lower right')
    plt.show()


def plotMADQLRewardChange(dataPath):
    plotRewardChange(dataPath, dataName="default-Algorithm.MADQL-averageCapacity", lineStyle="-")

    plt.ylabel("System Capacity (bps/Hz)")
    plt.xlabel("Time Slot")
    plt.show()


def plotRewardPenaltyPDF(dataPath, dataName):
    reward = loadData(dataPath, dataName)
    print(f"len of data: {len(reward)}")
    pdf(reward, label="log")

    plt.show()


if __name__ == "__main__":
    # plotDifferentAlpha("./simulation_data/data.txt")
    # plotMADQLRewardChange("./simulation_data/reward-data-003.txt")
    # plotMADQLRewardChange("./simulation_data/data.txt")
    # plotRewardPenaltyPDF("./simulation_data/data.txt", "RewardLogrewardPenalty")
    # plotRewardPenaltyPDF("./simulation_data/data.txt", "RewardSig-rewardPenalty")
    # plotMADQLRewardChange("./simulation_data/reward-data-008.txt")
    # plotCapacityCDFCompare("./simulation_data/data.txt")
    plot21LinkCDFCompare()
