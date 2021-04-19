import time
import logging
from utils import *

"""
def my_log(str):
    print(str)
    logging.info(str)

print(time.time(), type(time.time()))
print(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))

logging.basicConfig(filename="./log/logile.log", level=logging.DEBUG)
str = "lyc"
logging.info(f'hello {str}')
my_log(f'hello {str}')
"""


mylog = Logger()
str = "lyc"
mylog.log(f'Hello {str}!')
