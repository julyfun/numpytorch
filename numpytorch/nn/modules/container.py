import numpy as np
import os
from .module import Module


class Sequential(Module):
    def __init__(self, *args):
        super().__init__()

        # Define the layers of the neural network
        self.layers = [*args]
