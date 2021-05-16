import time
import logging
from hyper_parameter import *


class Logger:
    def __init__(self):
        # set path
        log_file_path = log_path + time.strftime("%Y-%m-%d") + ".log"
        logging.basicConfig(filename=log_file_path, format='%(message)s', level=logging.DEBUG)
        # starting write some information
        logging.info("-----------------------NETWORK CONFIG-------------------------------")
        logging.info(f'DATE: {time.strftime("%Y-%m-%d", time.localtime())}, '
                     f'TIME: {time.strftime("%H:%M:%S", time.localtime())}')
        logging.info(f'num_exp = {num_exp}, learning_rate = {learning_rate}\n'
                     f'num_of_iterations = {iteration}, mini_batch_size = {minibatch_size},\n'
                     f'layers = {nodes_num}')
        logging.info("-----------------------------END------------------------------------")

    def log(self, log_item):
        print(log_item)
        logging.info(log_item)
