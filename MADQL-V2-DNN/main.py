import matplotlib.pyplot as plt

from descision_maker import setDecisionMaker
from utils import setLogger, cdf, Algorithm, pdf
from config import *
from mobile_network import MobileNetwork, plotMobileNetwork
from plot_figure import plotRicianChannel, calIndicator, plotTempReportCapacity
from channel import Channel


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
        mn = MobileNetwork(loadNetwork="21-Links", totalTimeSlot=2000, printSlot=1)
        plotMobileNetwork(mn.getSectors(), mn.getUEs())

        mn.dm = setDecisionMaker(Algorithm.CELL_ES)
        mn.step()
        cdf(mn.getAverageCapacity(), label="CELL_ES")
        mn.clearRecord()

        plt.show()
    elif EXECUTION_MODE == "TRAIN_MADQL":
        mn = MobileNetwork(loadNetwork="21-Links", totalTimeSlot=5000, printSlot=50)
        plotMobileNetwork(mn.getSectors(), mn.getUEs())

        mn.dm = setDecisionMaker(Algorithm.MADQL)
        mn.step()

        cdf(mn.getAverageCapacity()[-2000:], label="MADQL")
        plt.show()
    elif EXECUTION_MODE == "TEST_MADQL":
        mn = MobileNetwork(loadNetwork="21-Links", trainNetwork=False, totalTimeSlot=2000, printSlot=10)
        plotMobileNetwork(mn.getSectors(), mn.getUEs())

        mn.dm = setDecisionMaker(Algorithm.MADQL, loadModel=True)
        mn.step()
    elif EXECUTION_MODE == "PRINT_SINGLE_CHANNEL_CSI":
        channel = Channel([0., 0., 10.], [10., 10., 1.5])
        print("CSI: ")
        print(channel.getCSI())
    elif EXECUTION_MODE == "PLOT_CHANNEL_PDF":
        plotRicianChannel()
    elif EXECUTION_MODE == "TEST_PLOT_PDF":
        gaussianData = np.random.normal(loc=0., scale=1., size=100000)
        pdf(gaussianData)
        plt.hist(gaussianData, color='blue', edgecolor='black', bins=2000)
        plt.show()
    elif EXECUTION_MODE == "PLOT_FINAL_PDF":
        plotTempReportCapacity(dataPath="./simulation_data/data.txt")
        calIndicator(dataPath="./simulation_data/data.txt")
