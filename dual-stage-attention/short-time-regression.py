import torch.nn as nn


class Simple1DCNN(nn.Module):
    """
    Implementation of a simple 1D CNN model.
    Dilation must be 2 so the output has the same length as the input.
    https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d
    """
    def __init__(self, T):
        super(Simple1DCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=T, out_channels=T, kernel_size=5, padding=4, dilation=2),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=T, out_channels=T, kernel_size=5, padding=4, dilation=2),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=T, out_features=T),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layer1(x)
        return x


class Group1DCNN(nn.Module): # TODO: Validade architecture. How input.shape = (T, 4) and output.shape = (T, 180)?
    """
    Pytorch example @ https://medium.com/@jiahao.cao.zh/trying-out-pytorchs-group-convolution-83cc270cdfd.
    """
    def __init__(self, T):
        super(Group1DCNN, self).__init__()
        self.layer1 = nn.Conv1d(in_channels=T, out_channels=T, groups=4, kernel_size=5, padding=4, dilation=2)

    def forward(self, x):
        x = self.layer1(x)
        return x