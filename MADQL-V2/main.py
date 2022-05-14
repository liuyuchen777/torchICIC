import matplotlib.pyplot as plt
import json

from descision_maker import setDecisionMaker
from utils import setLogger, cdf, Algorithm
from config import *
from mobile_network import MobileNetwork, plotMobileNetwork
from mobile_network_generator import saveMobileNetwork, loadMobileNetwork


def loadData(name="default"):
    with open(SIMULATION_DATA_PATH) as jsonFile:
        data = json.load(jsonFile)
    with open(SIMULATION_DATA_PATH, 'w') as jsonFile:
        return data[name]


def plotFigure():
    MADQL = loadData("Algorithm.MADQL-averageCapacity")
    Random = loadData("Algorithm.RANDOM-averageCapacity")

    cdf(MADQL, label="MADQL")
    cdf(Random, label="Random")

    plt.show()


if __name__ == "__main__":
    setLogger()
    controller = "RANDOM_AND_MAX_POWER"
    if controller == "RANDOM_AND_MAX_POWER":
        mn = MobileNetwork(loadNetwork="3-Links")

        plotMobileNetwork(mn.getSectors(), mn.getUEs())

        # mn.dm = setDecisionMaker(Algorithm.RANDOM)
        # mn.step()
        # cdf(mn.getAverageCapacity(), label="RANDOM")
        # mn.clearRecord()
        #
        # mn.dm = setDecisionMaker(Algorithm.MAX_POWER)
        # mn.step()
        # cdf(mn.getAverageCapacity(), label="MAX_POWER")
        # mn.clearRecord()

        # mn.dm = setDecisionMaker(Algorithm.CELL_ES)
        # mn.step()
        # cdf(mn.getAverageCapacity(), label="CELL_ES")
        # mn.clearRecord()
        #
        # plt.show()
        #
        mn.dm = setDecisionMaker(Algorithm.MADQL)
        mn.step()

        # plotFigure()
