import torch
import torch.nn as nn
from typing import List


class NeuralNet(nn.Module):
    """
    neural network class, with nn api
    """
    def __init__(self, input_size: int, hidden_size: List[int], output_size: int):
        """
        initialization function
        :param input_size: input data dimension
        :param hidden_size: list of hidden layer sizes
        :param output_size: output data dimension
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        """layers"""
        self.input = nn.Linear(self.input_size, self.hidden_size[0])
        for h in range(len(hidden_size) - 1):
            exec(f'self.hidden{h} = nn.Linear(self.hidden_size[h], self.hidden_size[h+1])')
        self.output = nn.Linear(hidden_size[-1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward propagation process, required by the nn.Module class
        :param x: the input data
        :return: the output from neural network
        """
        x = self.input(x)
        x = self.relu(x)
        for h in range(len(self.hidden_size) - 1):
            exec(f'z = self.hidden{h}(x)')
            x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        return x



