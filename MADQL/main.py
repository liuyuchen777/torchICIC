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
    plotMobileNetwork(mn)
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
    plotMobileNetwork(mn)
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

    saveMobileNetwork(mn)


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
    mn.saveRewards("MADQL_TRAIN_DECREASE")


def randomAndMADQL(mn):
    plotMobileNetwork(mn)

    # Eval MADQL
    mn.loadRewards("MADQL_TRAIN")
    averageRewards1 = mn.averageRewardRecord[500:]
    cdf(averageRewards1, label="MADQL")
    mn.cleanReward()

    # Random
    mn.loadRewards("RANDOM")
    averageRewards2 = mn.averageRewardRecord
    cdf(averageRewards2, label="Random")
    mn.cleanReward()

    # Cell ES
    mn.loadRewards("CELL_ES")
    averageRewards3 = mn.averageRewardRecord
    cdf(averageRewards3, label="Cell ES")
    mn.cleanReward()

    # plot
    plt.legend(loc='upper left')
    plt.show()


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
        mn = loadMobileNetwork()
        randAndCellES(mn)
    elif EXECUTION_MODE == "QUICK_TEST_MADQL":
        """[test] quick test MADQL"""
        mn = MobileNetwork()
        quickTestMQDQL(mn)
    elif EXECUTION_MODE == "TRAIN_MADQL":
        """[test] train MADQL model"""
        mn = loadMobileNetwork()
        trainMADQL(mn)
    elif EXECUTION_MODE == "EVAL_MADQL":
        mn = MobileNetwork()
        randomAndMADQL(mn)
