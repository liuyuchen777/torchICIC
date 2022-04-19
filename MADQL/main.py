from mobile_network import *
from utils import *
import matplotlib.pyplot as plt


def testSaveLoadMobileNetwork():
    """load mobile network parameter (position of UE, channel random variable) from local file"""
    # create and save
    mn = MobileNetwork()
    plotMobileNetwork(mn)
    saveMobileNetwork(mn)

    # load and plot
    mnNew = loadMobileNetwork()
    plotMobileNetwork(mnNew)


def randAndCellES(mn):
    """compare ES, random and plot reward"""
    # Cell ES
    mn.algorithm = Algorithm.CELL_ES
    mn.dm = setDecisionMaker(mn.algorithm)
    mn.train()
    averageRewards1 = mn.averageRewardRecord
    mn.saveRewards(name="CELL_ES")
    cdf(averageRewards1, label="Cell ES")

    # clean
    mn.cleanReward()

    # Random
    mn.algorithm = Algorithm.RANDOM
    mn.dm = setDecisionMaker(mn.algorithm)
    mn.train()
    averageRewards2 = mn.averageRewardRecord
    mn.saveRewards(name="RANDOM")
    cdf(averageRewards2, label="Random")

    plt.legend(loc='upper left')
    plt.show()


def randAndMaxPower(mn):
    # Random
    mn.algorithm = Algorithm.RANDOM
    mn.dm = setDecisionMaker(mn.algorithm)
    mn.train()
    averageRewards1 = mn.averageRewardRecord
    mn.saveRewards(name="RANDOM")
    cdf(averageRewards1, label="Random")

    # clean
    mn.cleanRewardRecord()

    # Max Power
    mn.algorithm = Algorithm.MAX_POWER
    mn.dm = setDecisionMaker(mn.algorithm)
    mn.train()
    averageRewards2 = mn.averageRewardRecord
    mn.saveRewards("Max_POWER")
    cdf(averageRewards2, label="Max Power")

    plt.legend(loc='upper left')
    plt.show()


def quickTestMQDQL(mn):
    mn.algorithm = Algorithm.MADQL
    mn.dm = setDecisionMaker(mn.algorithm)

    mn.train()

    mn.saveRewards("MADQL_TEST")


def trainMADQL(mn):
    plotMobileNetwork(mn)

    # MQDQL
    mn.algorithm = Algorithm.MADQL
    mn.dm = setDecisionMaker(mn.algorithm)
    mn.train()
    mn.saveRewards("MADQL_TRAIN")


if __name__ == "__main__":
    setLogger()
    EXECUTION_MODE = "TRAIN_MADQL"

    if EXECUTION_MODE == "NETWORK_STRUCTURE":
        """[test] plot network structure"""
        mn = MobileNetwork()
        plotMobileNetwork(mn)
    elif EXECUTION_MODE == "SAVE_AND_LOAD_MODEL":
        """[test] save and load mobile network"""
        testSaveLoadMobileNetwork()
    elif EXECUTION_MODE == "RAND_AND_MAX_POWER":
        """[test] random and max power"""
        mn = MobileNetwork()
        randAndMaxPower(mn)
    elif EXECUTION_MODE == "RAND_AND_CELL_ES":
        """[test] cell ES and random"""
        mn = MobileNetwork()
        randAndCellES(mn)
    elif EXECUTION_MODE == "QUICK_TEST_MADQL":
        """[test] quick test MADQL"""
        mn = MobileNetwork()
        quickTestMQDQL(mn)
    elif EXECUTION_MODE == "TRAIN_MADQL":
        """[test] train MADQL model"""
        mn = loadMobileNetwork()
        trainMADQL(mn)
