import torch


class MLP(torch.nn.Module):

    """Docstring for MLP. """

    def __init__(self, input_dim):
        """TODO: to be defined. """
        torch.nn.Module.__init__(self)
        self.layer1 = nn.Linear(input_dim, 256)
        self.tanh = nn.Tanh()
        self.layer2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.tanh(x)
        x = self.layer2(x)
        return x
