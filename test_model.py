import torch
from model_3d_beam_power import *

# simple load model
model = torch.load("../model/2021-04-14_17-09-54.pth")
model.eval()

# generate test data
# check how to use data_generator dir
