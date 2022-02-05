import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(
        self, num_layers, dropout, input_size, hidden_size, output_size,
        activate_function
    ):
        super().__init__()
        self.net = nn.Sequential()
        for i in range(num_layers-1):
            in_features = hidden_size if i != 0 else input_size
            self.net.add_module("layer_{}".format(i+1),
                                nn.Linear(in_features=in_features,
                                          out_features=hidden_size, ))
            self.net.add_module("activation_{}".format(i+1),
                                getattr(nn, activate_function)())
        self.net.add_module("layer_{}".format(num_layers),
                            nn.Linear(in_features=hidden_size, out_features=output_size))

    def forward(self, x):
        return self.net(x)


class LSTM(torch.nn.Module):

    def __init__(self, input_size, num_layers, dropout, hidden_size, output_size):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout, batch_first=True)
        # self.k = torch.nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, x):
        # x = torch.transpose(x, 0, 1)
        x, _ = self.lstm(x)
        x = torch.sum(x, dim=-1)
        # x = torch.transpose(x, 0, 1)
        return x
