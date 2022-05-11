import matplotlib.pyplot as plt

from descision_maker import setDecisionMaker
from utils import setLogger, cdf, Algorithm
from mobile_network import MobileNetwork, plotMobileNetwork


if __name__ == "__main__":
    setLogger()
    controller = "RANDOM_AND_MAX_POWER"
    if controller == "RANDOM_AND_MAX_POWER":
        mn = MobileNetwork()

        plotMobileNetwork(mn.getSectors(), mn.getUEs())

        mn.dm = setDecisionMaker(Algorithm.RANDOM)
        mn.step()
        cdf(mn.getAverageCapacity(), label="RANDOM")
        mn.clearRecord()

        mn.dm = setDecisionMaker(Algorithm.MAX_POWER)
        mn.step()
        cdf(mn.getAverageCapacity(), label="MAX_POWER")
        mn.clearRecord()

        # mn.dm = setDecisionMaker(Algorithm.CELL_ES)
        # mn.step()
        # cdf(mn.getAverageCapacity(), label="CELL_ES")
        # mn.clearRecord()

        plt.legend(loc='upper left')
        plt.show()
