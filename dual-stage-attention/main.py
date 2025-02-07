import torch
from torch.utils.data import DataLoader
from dataset import VideoDataset
from backbone import Backbone


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    backbone = Backbone().to(device)
    backbone.eval()

    inputs = [
        [a[0].to(device), a[1].to(device)] if type(a) == list else a.to(device)
        for a in next(iter(DataLoader(VideoDataset('./datasets/LIVE_NFLX_Plus'))))
    ]

    output = backbone(inputs)

    print('output.shape:', output.shape)


if __name__ == "__main__":
    main()