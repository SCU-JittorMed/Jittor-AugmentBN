# classification mnist example
import jittor as jt
from jittor import nn, Module
import numpy as np
import sys, os
import random
import math
from jittor import init

class Model (Module):
    def __init__ (self):
        super (Model, self).__init__()
        self.conv1 = nn.Conv (3, 32, 3, 1) # no padding
        self.conv2 = nn.Conv (32, 64, 3, 1)
        self.bn = nn.BatchNorm(64)

        self.max_pool = nn.Pool (2, 2)
        self.relu = nn.Relu()
        self.fc1 = nn.Linear (64 * 12 * 12, 256)
        self.fc2 = nn.Linear (256, 10)
    def execute (self, x) :
        x = self.conv1 (x)
        x = self.relu (x)

        x = self.conv2 (x)
        x = self.bn (x)
        x = self.relu (x)

        x = self.max_pool (x)
        x = jt.reshape (x, [x.shape[0], -1])
        x = self.fc1 (x)
        x = self.relu(x)
        x = self.fc2 (x)
        return x



