'''
Author: Liu Yuchen
Date: 2021-06-22 15:29:48
LastEditors: Liu Yuchen
LastEditTime: 2021-06-22 15:57:03
Description: A tool box for simulation
FilePath: /torch_ICIC/MADQL/utils.py
GitHub: https://github.com/liuyuchen777
'''

def dB2num(dB):
    num = 10 ** (dB / 10)
    return num


def num2dB(num):
    dB = 10 * np.log10(num)
    return dB

class Logger:
    def __init__(self, log_path="./log/"):
        # set path
        log_file_path = log_path + time.strftime("%Y-%m-%d") + ".log"
        logging.basicConfig(filename=log_file_path, format='%(message)s', level=logging.DEBUG)
        # starting write some information
        logging.info("-----------------------NETWORK CONFIG-------------------------------")
        # network config information
        
        logging.info("-----------------------------END------------------------------------")

    def log(self, log_item):
        print(log_item)
        logging.info(log_item)