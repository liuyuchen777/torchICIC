from MobileNetwork import *

import matplotlib.pyplot as plt


def testSaveLoadMobileNetwork():
    """load mobile network parameter (position of UE, channel random variable) from local file"""
    mn = MobileNetwork()
    mn.plotMobileNetwork()
    saveMobileNetwork(mn)
    mnNew = loadMobileNetwork()
    mnNew.plotMobileNetwork()


def testCellES():
    """compare ES, random and plot reward"""
    # load model
    mn = loadMobileNetwork()
    mn.config = Config()
    mn.cleanRewardRecord()
    mn.cleanAverageRewardRecord()

    # Cell ES
    mn.setDecisionMaker(Algorithm.CELL_ES)
    mn.train()
    averageRewards1 = mn.getAverageRewardRecord()
    mn.saveRewards(name="CELL_ES")
    cdf(averageRewards1, label="Cell ES")

    # clean
    mn.cleanRewardRecord()
    mn.cleanAverageRewardRecord()

    # Random
    mn.setDecisionMaker(Algorithm.RANDOM)
    mn.train()
    averageRewards2 = mn.getAverageRewardRecord()
    cdf(averageRewards2, label="Random")

    plt.show()


def showReward(mn):
    """"""
    # Random
    mn.setDecisionMaker(Algorithm.RANDOM)
    mn.train()
    averageRewards1 = mn.getAverageRewardRecord()
    cdf(averageRewards1, label="Random")

    mn.cleanRewardRecord()
    mn.cleanAverageRewardRecord()
    # Max Power
    mn.algorithm = Algorithm.MAX_POWER
    mn.dm = mn.setDecisionMaker(mn.algorithm)
    mn.train()
    averageRewards2 = mn.getAverageRewardRecord()
    cdf(averageRewards2, label="Max Power")

    plt.legend(loc='upper left')
    plt.show()


if __name__ == "__main__":
    setLogger()
    EXECUTION_MODE = "NETWORK_STRUCTURE"
    if EXECUTION_MODE == "NETWORK_STRUCTURE":
        """[test] network structure"""
        mn = MobileNetwork()
        mn.plotMobileNetwork()
    elif EXECUTION_MODE == "RAND_AND_MAX_POWER":
        """[test] reward in random and max power"""
        mn = MobileNetwork()
        showReward(mn)
    elif EXECUTION_MODE == "BUILD_STATE_AND_RECORD":
        """[test] build state and build record"""
        mn = loadMobileNetwork()
        mn.setConfig(Config())
        mn.setDecisionMaker(Algorithm.MADQL)
        mn.cleanReward()
        mn.train()
        mn.saveRewards("MADQL_Train")
    elif EXECUTION_MODE == "SAVE_AND_LOAD_MODEL":
        """[test] save and load mobile network"""
        testSaveLoadMobileNetwork()
    elif EXECUTION_MODE == "CELL_ES":
        """[test] cell ES"""
        testCellES()
