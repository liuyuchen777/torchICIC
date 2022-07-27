import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import savgol_filter
import numpy as np
from matplotlib.pyplot import MultipleLocator
import seaborn as sns

from utils import loadData, pdf, average, mid

matplotlib.rcParams.update({'font.size': 13})

"""Element Plot"""


def plotCapacityCDF(dataPath, dataName, dataNumber, *args, **kwargs):
    data = loadData(dataPath, dataName)
    data = data[-1 * dataNumber:]
    sns.ecdfplot(data=data, *args, **kwargs)


def plotCapacityCDFCenterThree(dataPath, dataName,  *args, **kwargs):
    data = loadData(dataPath, dataName)
    data = [sum(capacity[0:3]) / 3. for capacity in data]
    sns.ecdfplot(data=data, *args, **kwargs)


def plotCapacityCDFCenterThreeThreshold(dataPath, dataName, *args, **kwargs):
    data = loadData(dataPath, dataName)
    newData = []
    for capacity in data:
        temp = sum(capacity[0:3]) / 3.
        if temp < 0.32:
            continue
        newData.append(temp)
    sns.ecdfplot(data=newData, *args, **kwargs)


def calAndPrintIndicator(dataPath, dataName):
    data = loadData(path=dataPath, name=dataName)
    print(f"{dataName} average: {average(data)}, mid: {mid(data)}")


def calAndPrintIndicatorCenterThree(dataPath, dataName):
    data = loadData(path=dataPath, name=dataName)
    data = [sum(capacity[0:3]) / 3. for capacity in data]
    print(f"{dataName} average: {average(data)}, mid: {mid(data)}")


def plotRewardChange(dataPath, dataName, *args, **kwargs):
    data = loadData(path=dataPath, name=dataName)

    data = savgolAverage(data, windowLen=701, polyOrder=5)
    timeSlot = range(len(data))

    plt.plot(timeSlot, data, *args, **kwargs)


def plotRewardChangeCenterThree(dataPath, dataName, *args, **kwargs):
    data = loadData(path=dataPath, name=dataName)
    data = [sum(capacity[0:3]) / 3. for capacity in data]

    data = savgolAverage(data, windowLen=701, polyOrder=5)
    timeSlot = range(len(data))

    plt.plot(timeSlot, data, *args, **kwargs)


def savgolAverage(data, windowLen, polyOrder):
    return savgol_filter(data, window_length=windowLen, polyorder=polyOrder)


def windowAverage(data, N):
    """average on previous N time slot"""
    averageData = []
    for i in range(len(data)):
        sumStart = i - N if (i - N) > 0 else 0
        sumLen = N if (i - N) > 0 else i + 1
        averageData.append(sum(data[sumStart:i + 1]) / sumLen)
    return averageData


"""Aggregation Plot"""


def plotDifferentAlpha(dataPath):
    plotCapacityCDF(dataPath, dataName="alpha0.01-Algorithm.MADQL-averageCapacity", dataNumber=1000,
                    label="Alpha=0.01", linestyle="-")
    plotCapacityCDF(dataPath, dataName="alpha0.1-Algorithm.MADQL-averageCapacity", dataNumber=1000,
                    label="Alpha=0.1", linestyle="-")
    plotCapacityCDF(dataPath, dataName="alpha10-Algorithm.MADQL-averageCapacity", dataNumber=1000,
                    label="Alpha=1.0", linestyle=":")
    plotCapacityCDF(dataPath, dataName="alpha5-Algorithm.MADQL-averageCapacity", dataNumber=1000,
                    label="Alpha=5.0", linestyle=":")
    plotCapacityCDF(dataPath, dataName="alpha1-Algorithm.MADQL-averageCapacity", dataNumber=1000,
                    label="Alpha=10.0", linestyle="-.")

    plt.xlabel("Average System Capacity (bps/Hz)")
    plt.ylabel("CDF")
    plt.legend(loc='upper left')
    plt.show()


def plot3LinkCDFCompare():
    plotCapacityCDF("./simulation_data/reward-data-009.txt", dataName="3-Links-Test-Algorithm.MADQL-averageCapacity",
                    dataNumber=2000,
                    label="MADQL", linestyle="-")
    plotCapacityCDF("./simulation_data/reward-data-009.txt", dataName="default-Algorithm.RANDOM-averageCapacity",
                    dataNumber=2000,
                    label="Random", linestyle="--")
    plotCapacityCDF("./simulation_data/reward-data-009.txt", dataName="power-Algorithm.CELL_ES-averageCapacity",
                    dataNumber=2000,
                    label="Power ES", linestyle="-.")
    plotCapacityCDF("./simulation_data/reward-data-009.txt", dataName="beam-Algorithm.CELL_ES-averageCapacity",
                    dataNumber=2000,
                    label="Beam ES", linestyle=":")
    plotCapacityCDF("./simulation_data/reward-data-009.txt", dataName="default-Algorithm.CELL_ES-averageCapacity",
                    dataNumber=2000,
                    label="Joint ES", linestyle=":")

    calAndPrintIndicator("./simulation_data/reward-data-009.txt", dataName="3-Links-Test-Algorithm.MADQL-averageCapacity")
    calAndPrintIndicator("./simulation_data/reward-data-009.txt", dataName="default-Algorithm.RANDOM-averageCapacity")
    calAndPrintIndicator("./simulation_data/reward-data-009.txt", dataName="power-Algorithm.CELL_ES-averageCapacity")
    calAndPrintIndicator("./simulation_data/reward-data-009.txt", dataName="beam-Algorithm.CELL_ES-averageCapacity")
    calAndPrintIndicator("./simulation_data/reward-data-009.txt", dataName="default-Algorithm.CELL_ES-averageCapacity")

    plt.xlabel("Average System Capacity (bps/Hz)")
    plt.ylabel("CDF")
    plt.legend(loc='lower right')
    plt.show()


def plot21LinkCDFCompare():
    matplotlib.rcParams.update({'font.size': 13})

    plotCapacityCDFCenterThreeThreshold("./simulation_data/reward-data-013.txt", dataName="default-Algorithm.MADQL-capacity",
                                        label="MADQL", linestyle="-")
    plotCapacityCDFCenterThree("./simulation_data/reward-data-013.txt", dataName="default-Algorithm.RANDOM-capacity",
                               label="Random", linestyle="--")
    plotCapacityCDFCenterThree("./simulation_data/reward-data-017.txt", dataName="default-Algorithm.CELL_ES-capacity",
                               label="Local ES", linestyle="-.")

    calAndPrintIndicatorCenterThree("./simulation_data/reward-data-013.txt", dataName="default-Algorithm.MADQL-capacity")
    calAndPrintIndicatorCenterThree("./simulation_data/reward-data-013.txt", dataName="default-Algorithm.RANDOM-capacity")
    calAndPrintIndicatorCenterThree("./simulation_data/reward-data-017.txt", dataName="default-Algorithm.CELL_ES-capacity")

    plt.xlabel("Average System Capacity (bps/Hz)")
    plt.ylabel("CDF")
    plt.legend(loc='lower right')
    plt.show()


def plotDifferentAlphaRewardChangeCenterThree():
    """this may take long time"""

    dataPath = "./simulation_data/reward-data-010.txt"

    plotRewardChangeCenterThree(dataPath, dataName="alpha0.01-Algorithm.MADQL-capacity",
                    label="Alpha=0.01", linestyle="-")
    plotRewardChangeCenterThree(dataPath, dataName="alpha0.1-Algorithm.MADQL-capacity",
                    label="Alpha=0.1", linestyle="-")
    plotRewardChangeCenterThree(dataPath, dataName="alpha10-Algorithm.MADQL-capacity",
                    label="Alpha=1", linestyle=":")
    plotRewardChangeCenterThree(dataPath, dataName="alpha5-Algorithm.MADQL-capacity",
                    label="Alpha=5", linestyle=":")
    plotRewardChangeCenterThree(dataPath, dataName="alpha1-Algorithm.MADQL-capacity",
                    label="Alpha=10", linestyle="-.")

    plt.ylabel("System Capacity (bps/Hz)")
    plt.xlabel("Time Slot")
    plt.legend()
    plt.show()


def plotDifferentAlphaRewardChange():
    """this may take long time"""

    dataPath = "./simulation_data/reward-data-010.txt"

    plotRewardChange(dataPath, dataName="alpha0.01-Algorithm.MADQL-averageCapacity",
                                label="Alpha=0.01", linestyle="-")
    plotRewardChange(dataPath, dataName="alpha0.1-Algorithm.MADQL-averageCapacity",
                                label="Alpha=0.1", linestyle="-")
    plotRewardChange(dataPath, dataName="alpha10-Algorithm.MADQL-averageCapacity",
                                label="Alpha=1", linestyle=":")
    plotRewardChange(dataPath, dataName="alpha5-Algorithm.MADQL-averageCapacity",
                                label="Alpha=5", linestyle=":")
    plotRewardChange(dataPath, dataName="alpha1-Algorithm.MADQL-averageCapacity",
                                label="Alpha=10", linestyle="-.")

    plt.ylabel("System Capacity (bps/Hz)")
    plt.xlabel("Time Slot")
    plt.legend()
    plt.show()


def plotMADQL3LinkRewardChange():
    plotRewardChangeCenterThree(dataPath="./simulation_data/reward-data-008.txt",
                                dataName="default-Algorithm.MADQL-capacity", label="MADQL")

    plt.ylabel("System Capacity (bps/Hz)")
    plt.xlabel("Time Slot")
    plt.legend()
    plt.show()


def plotMADQL21LinkRewardChange():
    plotRewardChangeCenterThree(dataPath="./simulation_data/reward-data-012.txt",
                                dataName="50000-link21-alpha5-Algorithm.MADQL-capacity", label="MADQL")

    plt.ylabel("System Capacity (bps/Hz)")
    plt.xlabel("Time Slot")
    plt.legend()
    plt.show()


def plotRewardPenaltyPDF(dataPath, dataName):
    reward = loadData(dataPath, dataName)
    print(f"len of data: {len(reward)}")
    pdf(reward, label="log")

    plt.show()


def calLinkAverage(data):
    res = [0. for _ in range(len(data[0]))]
    for row in data:
        for index in range(len(row)):
            res[index] += row[index]
    res = [link / len(data) for link in res]
    return res


def plotLinksAverageCapacity21Link():

    plt.figure(figsize=(15, 8))

    madql = loadData("./simulation_data/reward-data-013.txt", "default-Algorithm.MADQL-capacity")
    es = loadData("./simulation_data/reward-data-017.txt", "default-Algorithm.CELL_ES-capacity")
    random = loadData("./simulation_data/reward-data-013.txt", "default-Algorithm.RANDOM-capacity")

    madql = calLinkAverage(madql)
    es = calLinkAverage(es)
    random = calLinkAverage(random)

    x = np.arange(len(madql))
    total_width, n = 0.8, 3
    width = total_width / n
    x = x - (total_width - width) / 2

    plt.bar(x, random, width=width, label="Random")
    plt.bar(x + width, es, width=width, label="Local ES")
    plt.bar(x + 2 * width, madql, width=width, label="MADQL")

    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(-0.5, 20.5)

    plt.ylabel("Link Capacity (bps/Hz)")
    plt.xlabel("Link Index")
    plt.legend()
    plt.show()


def plotLinksAverageCapacity3Link():
    madql = loadData("./simulation_data/reward-data-016.txt", "default-Algorithm.MADQL-capacity")
    es = loadData("./simulation_data/reward-data-009.txt", "default-Algorithm.CELL_ES-capacity")
    random = loadData("./simulation_data/reward-data-009.txt", "default-Algorithm.RANDOM-capacity")

    madql = calLinkAverage(madql)
    es = calLinkAverage(es)
    random = calLinkAverage(random)

    x = np.arange(len(madql))
    total_width, n = 0.8, 3
    width = total_width / n
    x = x - (total_width - width) / 2

    plt.bar(x, random, width=width, label="Random")
    plt.bar(x + width, es, width=width, label="Local ES")
    plt.bar(x + 2 * width, madql, width=width, label="MADQL")

    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)

    plt.ylabel("Link Capacity (bps/Hz)")
    plt.xlabel("Link Index")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # plotDifferentAlpha("./simulation_data/reward-data-011.txt")
    plot21LinkCDFCompare()
    # plot3LinkCDFCompare()
    # plotLinksAverageCapacity21Link()
    # plotLinksAverageCapacity3Link()
    # plotMADQL21LinkRewardChange()
    # plotMADQL3LinkRewardChange()
    # plotDifferentAlphaRewardChange()
