from MobileNetwork import *


def showReward(mn):
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


def testSaveLoadMobileNetwork():
    mn = MobileNetwork()
    mn.plotMobileNetwork()
    saveMobileNetwork(mn)
    mnNew = loadMobileNetwork()
    mnNew.plotMobileNetwork()


def testCellES():
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


def drawMidTerm():
    # load model
    mn = loadMobileNetwork()

    # CELL ES
    mn.loadRewards("CELL_ES")
    averageRewards = mn.getAverageRewardRecord()
    cdf(averageRewards, label="Cell ES")

    # Max Power
    mn.loadRewards("MaxPower")
    averageRewards3 = mn.getAverageRewardRecord()
    cdf(averageRewards3, label="Max Power")

    # Random
    mn.loadRewards("Random")
    averageRewards2 = mn.getAverageRewardRecord()
    cdf(averageRewards2, label="Random")

    # MADQL
    mn.loadRewards("MADQL")
    averageRewards4 = mn.getAverageRewardRecord()
    cdf(averageRewards4, label="MADQL")

    plt.ylabel("Empirical Cumulative Probability")
    plt.xlabel("Average Spectral Efficiency (bps/Hz) per Link")
    plt.title("Empirical CDF")
    plt.legend(loc='upper left')
    plt.show()


if __name__ == "__main__":
    setLogger()
    """[test] network structure"""
    # mn = MobileNetwork()
    # mn.plotMobileNetwork()
    """[test] reward in random and max power"""
    # mn = MobileNetwork()
    # showReward(mn)
    """[test] build state and build record"""
    mn = loadMobileNetwork()
    mn.setConfig(Config())
    mn.setDecisionMaker(Algorithm.MADQL)
    mn.cleanReward()
    mn.train()
    mn.saveRewards("MADQL_Train")
    """[test] save and load mobile network"""
    # testSaveLoadMobileNetwork()
    """[test] cell ES"""
    # testCellES()
    """[test] mid term"""
    # drawMidTerm()
