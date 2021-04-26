import time
import logging
from utils import *
from torch import nn
import torch

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

loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
label = torch.argmax(input, dim=1)
target = torch.empty(3, dtype=torch.long).random_(5)
print(label)
print(target)
output = loss(input, target)
print(output)
