import torch
from torch.utils.data import DataLoader

from src.dataset import ReviewDataset

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class ReviewDataLoader(DataLoader):
    def __init__(self,
                 dataset: ReviewDataset,
                 batch_size: int,
                 shuffle: bool = True):
        super().__init__(dataset, batch_size, shuffle)

    def __iter__(self) -> tuple[list[str], torch.Tensor]:
        for features, labels in super().__iter__():
            features, labels = list(features[0]), labels[0].tolist()
            batch = {"text": features, "label": labels}
            yield batch
