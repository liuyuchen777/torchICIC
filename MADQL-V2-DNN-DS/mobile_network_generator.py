import json
import logging

import matplotlib.pyplot as plt

from ue import UE
from sector import Sector
from config import *


SAME_DISTRIBUTION = False

radius_random_seed = np.random.rand()
angle_random_seed = np.random.rand()


def generateSector(centerIndex, centerX, centerY):
    sectors = []
    r = CELL_SIZE
    h = BS_HEIGHT
    sectors.append(Sector(centerIndex * 3, [centerX - r / 2, centerY - r / 2 * np.sqrt(3), h]))
    sectors.append(Sector(centerIndex * 3 + 1, [centerX + r, centerY, h]))
    sectors.append(Sector(centerIndex * 3 + 2, [centerX - r / 2, centerY + r / 2 * np.sqrt(3), h]))

    return sectors


def generateUE(centerIndex, sectors):
    UEs = []
    R = R_MAX - R_MIN
    h = UT_HEIGHT
    for sector, i in zip(sectors, range(3)):
        # generate r and theta
        if SAME_DISTRIBUTION:
            r = R * radius_random_seed + R_MIN
            theta = (angle_random_seed * 120 + 120 * i) / 360 * 2 * np.pi
        else:
            r = R * np.random.rand() + R_MIN
            theta = (np.random.rand() * 120 + 120 * i) / 360 * 2 * np.pi
        # r-theta to x-y
        posX = sector.getPosition()[0] + r * np.cos(theta)
        posY = sector.getPosition()[1] + r * np.sin(theta)
        # append UE in UEs
        UEs.append(UE(centerIndex * 3 + i, [posX, posY, h]))

    return UEs


def generateMobileNetwork():
    sectors = []
    UEs = []

    for i in range(CELL_NUMBER):
        if i == 0:
            centerX = 0.
            centerY = 0.
        else:
            centerR = CELL_SIZE * np.sqrt(3)
            centerAngle = (-150 + (i - 1) * 60) / 360 * 2 * np.pi
            centerX = centerR * np.cos(centerAngle)
            centerY = centerR * np.sin(centerAngle)
        # generate sector and ue position
        tmpSectors = generateSector(i, centerX, centerY)
        tmpUEs = generateUE(i, tmpSectors)
        sectors.extend(tmpSectors)
        UEs.extend(tmpUEs)
    logging.getLogger().info(f"--------------------Create New Mobile Network------------------")
    return sectors, UEs


def plotMobileNetwork(sectors, UEs):
    for i in range(CELL_NUMBER):
        if i == 0:
            centerX = 0.
            centerY = 0.
        else:
            centerR = CELL_SIZE * np.sqrt(3)
            centerAngle = (-150 + (i - 1) * 60) / 360 * 2 * np.pi
            centerX = centerR * np.cos(centerAngle)
            centerY = centerR * np.sin(centerAngle)
        tmpSectors = [sectors[i * 3], sectors[i * 3 + 1], sectors[i * 3 + 2]]
        tmpUEs = [UEs[i * 3], UEs[i * 3 + 1], UEs[i * 3 + 2]]
        plotCell(centerX, centerY, tmpSectors, tmpUEs)

    plt.xlabel("x/m")
    plt.ylabel("y/m")
    plt.title("3-Links Mobile Network")
    plt.show()


def plotCell(centerX, centerY, sectors, UEs):
    sectorsPosX = []
    sectorsPosY = []
    UEsPosX = []
    UEsPosY = []
    cellSize = CELL_SIZE
    for sector, UE in zip(sectors, UEs):
        sectorsPosX.append(sector.getPosition()[0])
        sectorsPosY.append(sector.getPosition()[1])
        UEsPosX.append(UE.getPosition()[0])
        UEsPosY.append(UE.getPosition()[1])
    # plot point
    plt.scatter(sectorsPosX, sectorsPosY, c='r')
    plt.scatter(UEsPosX, UEsPosY, c='b')
    # draw Hexagon
    theta = np.linspace(0, 2 * np.pi, 13)
    x = np.cos(theta)
    x[1::2] *= 0.5
    y = np.sin(theta)
    y[1::2] *= 0.5
    plt.plot(x[::2] * cellSize + centerX, y[::2] * cellSize + centerY, color='r')
    # print sector line
    point = np.linspace([-cellSize, 0], [0, 0], 10) + [centerX, centerY]
    plt.plot(point[:, 0], point[:, 1], 'k--')
    point = np.linspace([0, 0], [cellSize / 2, cellSize / 2 * np.sqrt(3)], 10) + [centerX, centerY]
    plt.plot(point[:, 0], point[:, 1], 'k--')
    point = np.linspace([0, 0], [cellSize / 2, - cellSize / 2 * np.sqrt(3)], 10) + [centerX, centerY]
    plt.plot(point[:, 0], point[:, 1], 'k--')


def loadMobileNetwork(name="default"):
    with open(MOBILE_NETWORK_DATA_PATH) as jsonFile:
        data = json.load(jsonFile)
        UEPositions = data[name + "-UE-positions"]
        sectorPositions = data[name + "-sector-positions"]
        sectors = []
        UEs = []
        for i in range(len(sectorPositions)):
            sectors.append(Sector(i, sectorPositions[i]))
            UEs.append(UE(i, UEPositions[i]))
    logging.getLogger().info(f"--------------------Load Mobile Network {name}------------------")
    return sectors, UEs


def saveMobileNetwork(sectors, UEs, name="default"):
    sectorPositions = []
    UEPositions = []
    for sector, UE in zip(sectors, UEs):
        sectorPositions.append(sector.getPosition())
        UEPositions.append(UE.getPosition())
    with open(MOBILE_NETWORK_DATA_PATH) as jsonFile:
        data = json.load(jsonFile)
    with open(MOBILE_NETWORK_DATA_PATH, 'w') as jsonFile:
        data[name + "-UE-positions"] = UEPositions
        data[name + "-sector-positions"] = sectorPositions
        json.dump(data, jsonFile)
    logging.getLogger().info(f"--------------------Save Mobile Network {name}------------------")