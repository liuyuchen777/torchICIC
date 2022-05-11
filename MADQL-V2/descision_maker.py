from utils import Algorithm
from random_dm import Random
from cell_es_dm import CellES
from madql_dm import MADQL
from max_power_dm import MaxPower


def setDecisionMaker(algorithm, loadModel=False):
    """
    Return instance of decision maker base on algorithm
    Args:
        loadModel: create new NN or load exist NN
        algorithm:

    Returns:
        decision maker
    """
    if algorithm == Algorithm.RANDOM:
        return Random()
    elif algorithm == Algorithm.MAX_POWER:
        return MaxPower()
    elif algorithm == Algorithm.MADQL:
        return MADQL(loadModel)
    elif algorithm == Algorithm.CELL_ES:
        return CellES()
    else:
        raise Exception("Incorrect algorithm setting: " + algorithm)