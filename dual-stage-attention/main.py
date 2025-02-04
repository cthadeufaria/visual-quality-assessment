import torch
from torch.utils.data import DataLoader
from dataset import ExampleDataset
from backbone import Backbone


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    backbone = Backbone().to(device)
    backbone.eval()

    inputs = [i.to(device) for i in next(iter(DataLoader(ExampleDataset())))]
    output = backbone(inputs)

    print('output.shape:', output.shape)


if __name__ == "__main__":
    main()