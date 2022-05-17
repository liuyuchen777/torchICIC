import matplotlib.pyplot as plt
import json

from descision_maker import setDecisionMaker
from utils import setLogger, cdf, Algorithm, pdf
from config import *
from mobile_network import MobileNetwork, plotMobileNetwork
from mobile_network_generator import saveMobileNetwork, loadMobileNetwork
from plot_figure import plotFinalReportCapacity, plotRicianChannel
from channel import Channel

if __name__ == "__main__":
    setLogger()
    EXECUTION_MODE = "CHANNEL_PDF"
    if EXECUTION_MODE == "RANDOM_AND_MAX_POWER":
        # mn = MobileNetwork()

        # plotMobileNetwork(mn.getSectors(), mn.getUEs())

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
        # mn.dm = setDecisionMaker(Algorithm.MADQL)
        # mn.step()

        plotFinalReportCapacity()
    elif EXECUTION_MODE == "SINGLE_CHANNEL":
        """[test] single channel value"""
        channel = Channel([0., 0., 10.], [10., 10., 1.5])
        print("CSI: ")
        print(channel.getCSI())
    elif EXECUTION_MODE == "CHANNEL_PDF":
        plotRicianChannel()
    elif EXECUTION_MODE == "TEST_PLOT_PDF":
        """[test] pdf function work or not"""
        gaussianData = np.random.normal(loc=0., scale=1., size=100000)
        pdf(gaussianData)
        plt.hist(gaussianData, color='blue', edgecolor='black', bins=2000)
        plt.show()
