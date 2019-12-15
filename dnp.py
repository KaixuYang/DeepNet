import torch
import numpy as np
from torch import nn
from typing import List
from warnings import warn
import copy
import pandas as pd


class NeuralNet(nn.Module):
    """
    neural network class, with nn api
    """
    def __init__(self, input_size: int, hidden_size: List[int], output_size: int):
        """
        initialization function
        :param input_size: input data dimension
        :param hidden_size: list of hidden layer sizes, arbitrary length
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
        self.hiddens = nn.ModuleList([nn.Linear(self.hidden_size[h], self.hidden_size[h + 1])for h in range(len(self.hidden_size) - 1)])
        self.output = nn.Linear(hidden_size[-1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward propagation process, required by the nn.Module class
        :param x: the input data
        :return: the output from neural network
        """
        x = self.input(x)
        x = self.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        return x


class DeepNet:
    """
    implements the deep neural network in "https://www.ijcai.org/proceedings/2017/0318.pdf"
    """
    def __init__(self, max_feature: int, num_classes: int = 2, hidden_size: list = None, gpu: bool = True,
                 q: int = 2, num_dropout: int = 10):
        """
        initialization function
        """
        self.gpu = gpu
        """model parameters"""
        self.max_feature = max_feature
        self.n = None
        self.p = None
        self.q = q  # norm used in feature selection
        self.num_classes = num_classes
        if hidden_size is None:
            self.hidden_size = [50]
        else:
            self.hidden_size = hidden_size
        self.num_dropout = num_dropout
        """candidate sets"""
        self.S = None
        self.C = None
        self.selected = None
        """model"""
        self.nnmodel = None
        """optimization parameter"""
        self.criterion = None
        self.learning_rate = None
        self.batch_size = None
        self.epochs = None
        self.dropout_prop = None

    @staticmethod
    def add_bias(x: torch.Tensor) -> torch.Tensor:
        if not all(x[:, 0] == 1):
            x = torch.cat((torch.ones(x.shape[0], 1), x.float()), dim=1)
        return x

    def initialize(self):
        """
        initialize parameters
        """
        self.S = {0}
        self.selected = {}
        self.C = set(range(1, self.p))

    def set_parameters(self, x: torch.Tensor, batch_size: int, learning_rate: float, epochs: int,
                       dropout_prop: list = None):
        """
        set optimization parameters in class
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        self.n, self.p = x.shape
        if dropout_prop is None:
            self.dropout_prop = [0.5] * len(self.hidden_size)
        else:
            self.dropout_prop = dropout_prop

    def update_nn_weight(self, x: torch.Tensor, y: torch.Tensor):
        """
        updates the weights of neural network with the selected inputs
        """
        input_size = len(self.S)
        hidden_size = self.hidden_size
        output_size = self.num_classes
        self.nnmodel = NeuralNet(input_size, hidden_size, output_size)
        x = x[:, list(self.S)]
        optimizer = torch.optim.Adam(self.nnmodel.parameters(), lr=self.learning_rate)
        trainset = []
        for i in range(x.shape[0]):
            trainset.append([x[i, :], y[i]])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        for e in range(self.epochs):
            running_loss = 0
            for data, label in trainloader:
                input_0 = data.view(data.shape[0], -1)
                optimizer.zero_grad()
                output = self.nnmodel(input_0.float())
                loss = self.criterion(output, label.squeeze(1))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            else:
                print(f"Epoch: {e + 1}\nTraining loss: {running_loss/len(trainloader)}")

    def dropout(self) -> nn.Module:
        """
        returns a copy of randomly dropout model
        """
        if len([p for p in self.dropout_prop if p >= 1 or p <= 0]) > 0:
            raise ValueError("Dropout proportion must be between 0 and 1.")
        if len(self.dropout_prop) > len(self.hidden_size):
            warn(f"Too many dropout proportions, only the first {len(self.hidden_size)} will be used")
        elif len(self.dropout_prop) < len(self.hidden_size):
            warn(f"Too few dropout proportions, dropout won't be applied to the last "
                 f"{len(self.hidden_size) - len(self.dropout_prop) - 1} layers")
        model_dp = copy.deepcopy(self.nnmodel)
        for h in range(min(len(self.hidden_size) - 1, len(self.dropout_prop))):
            prop = self.dropout_prop[h]
            h_size = self.hidden_size[h]
            dropout_index = np.random.choice(range(h_size), int(h_size * prop), replace=False)
            model_dp.hiddens[h].weight[:, dropout_index] = torch.zeros(model_dp.hiddens[h].weight[:, dropout_index].shape)
        if len(self.hidden_size) <= len(self.dropout_prop):
            prop = self.dropout_prop[len(self.hidden_size) - 1]
            h_size = self.hidden_size[-1]
            dropout_index = np.random.choice(range(h_size), int(h_size * prop), replace=False)
            model_dp.output.weight[:, dropout_index] = torch.zeros(model_dp.output.weight[:, dropout_index].shape)
        return model_dp

    def compute_input_gradient(self, x: torch.Tensor, y: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """
        computes the input gradients given a model after dropout
        """
        model_gr = NeuralNet(self.p, self.hidden_size, self.num_classes)
        temp = torch.zeros(model_gr.input.weight.shape)
        temp[:, list(self.S)] = model.input.weight
        model_gr.input.weight.data = temp
        for h in range(len(self.hidden_size) - 1):
            model_gr.hiddens[h].weight.data = model.hiddens[h].weight
        model_gr.output.weight.data = model.output.weight
        output_gr = model_gr(x.float())
        loss_gr = self.criterion(output_gr, y.squeeze(1))
        loss_gr.backward()
        input_gradient = model_gr.input.weight.grad
        return input_gradient

    def average_input_gradient(self, x: torch.Tensor, y: torch.Tensor, num_average: int):
        """
        compute the average input gradient over different dropouts
        """
        grad_cache = None
        for num in range(num_average):
            model = self.dropout()
            input_grad = self.compute_input_gradient(x, y, model)
            if grad_cache is None:
                grad_cache = input_grad
            else:
                grad_cache += input_grad
        return grad_cache / num_average

    def find_next_input(self, input_gradient: torch.Tensor) -> int:
        """
        computes which input is to be selected next by finding the maximum gradient norm
        """
        gradient_norm = input_gradient.norm(p=self.q, dim=0)
        gradient_norm[list(self.S)] = 0
        max_index = torch.argmax(gradient_norm)
        return max_index.item()

    def update_sets(self, max_index):
        """
        updates the selected set and candidate set
        """
        self.S.add(max_index)
        self.C.remove(max_index)
        self.selected = self.S.copy()
        self.selected.remove(0)

    def numpy_to_torch(self, x):
        """
        convert array or dataframe to torch tensor
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        elif isinstance(x, pd.DataFrame):
            x = torch.from_numpy(x.values)
        return x

    def train(self, x: torch.Tensor, y: torch.Tensor, batch_size: int = 100, learning_rate: float = 0.005,
              epochs: int = 20, dropout_prop: list = None):
        """
        train the deep neural network on x and y
        """
        x = self.numpy_to_torch(x)
        y = self.numpy_to_torch(y)
        x = self.add_bias(x)
        """set parameters"""
        self.set_parameters(x, batch_size, learning_rate, epochs, dropout_prop)
        """initialization"""
        self.initialize()
        """start feature selection"""
        while len(self.S) < self.max_feature + 1:
            self.update_nn_weight(x, y)
            input_gradient = self.average_input_gradient(x, y, self.num_dropout)
            max_index = self.find_next_input(input_gradient)
            self.update_sets(max_index)
            print(f"Number of features selected is {len(self.S) - 1}.")
        self.update_nn_weight(x, y)
        print(f"Feature selection completed, selected features are {self.selected}")

    def predict(self, x) -> torch.Tensor:
        """
        making prediction with the
        """
        if self.nnmodel is None:
            raise ValueError("Model not trained, please run train first.")
        x = self.numpy_to_torch(x)
        x = self.add_bias(x)
        if x.shape[1] != self.p:
            raise ValueError("Dimension of x is wrong.")
        x = x[:, list(self.S)].float()
        y_pred = self.nnmodel(x)
        y_pred = torch.argmax(y_pred, dim=1)
        return y_pred

    def predict_error(self, x, y):
        """
        makes prediction on x and computes the prediction error from y
        """
        y_pred = self.predict(x)
        y_pred = y_pred.squeeze().float()
        y = self.numpy_to_torch(y)
        y = y.squeeze().float()
        acc = 1 - (abs(y - y_pred)).mean().item()
        print(f"Testing accuracy is {acc}.")
