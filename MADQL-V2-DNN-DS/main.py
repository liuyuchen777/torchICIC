import matplotlib.pyplot as plt

from descision_maker import setDecisionMaker
from utils import setLogger, cdf, Algorithm
from mobile_network import MobileNetwork, plotMobileNetwork


if __name__ == "__main__":
    setLogger()
    EXECUTION_MODE = "TRAIN_MADQL"

    if EXECUTION_MODE == "TEST_RANDOM":
        mn = MobileNetwork(loadNetwork="21-Links", totalTimeSlot=2000, printSlot=10)
        plotMobileNetwork(mn.getSectors(), mn.getUEs())

        mn.dm = setDecisionMaker(Algorithm.RANDOM)
        mn.step()
        cdf(mn.getAverageCapacity(), label="RANDOM")
        mn.clearRecord()

        plt.show()
    elif EXECUTION_MODE == "TEST_CELL_ES":
        mn = MobileNetwork(loadNetwork="3-Links", totalTimeSlot=2000, printSlot=1)
        plotMobileNetwork(mn.getSectors(), mn.getUEs())

        mn.dm = setDecisionMaker(Algorithm.CELL_ES)
        mn.step()
        cdf(mn.getAverageCapacity(), label="CELL_ES")
        mn.clearRecord()

        plt.show()
    elif EXECUTION_MODE == "TRAIN_MADQL":
        prefix = "RewardLog"
        mn = MobileNetwork(loadNetwork="21-Links", totalTimeSlot=1000, printSlot=50, savePrefix=prefix)
        plotMobileNetwork(mn.getSectors(), mn.getUEs())

        mn.dm = setDecisionMaker(Algorithm.MADQL)
        mn.step()
    elif EXECUTION_MODE == "TEST_MADQL":
        prefix = "test"
        mn = MobileNetwork(loadNetwork="21-Links", trainNetwork=False, totalTimeSlot=2000, printSlot=10, savePrefix=prefix)
        plotMobileNetwork(mn.getSectors(), mn.getUEs())

        mn.dm = setDecisionMaker(Algorithm.MADQL, loadModel=True)
        mn.step()
