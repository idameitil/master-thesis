import numpy as np
import sys
import subprocess
import os

def early_stopping():

    for partition in partitions:
        validation = partition
        train = partitions.remove(partition)
        for i in range(4):
            # train on train[i]
            # test on validation

