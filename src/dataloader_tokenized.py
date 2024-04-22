import torch
from torch.utils.data import DataLoader

from src.dataset import ReviewDataset
from src.tokenizer import Tokenizer

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class ReviewDataLoader(DataLoader):
    def __init__(self,
                 dataset: ReviewDataset,
                 tokenizer: Tokenizer,
                 batch_size: int,
                 shuffle: bool = True):
        super().__init__(dataset, batch_size, shuffle)
        self.tokenizer = tokenizer

    def __iter__(self) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        for features, labels in super().__iter__():
            features, labels = list(features[0]), labels[0]
            yield self.tokenizer(features), labels.long().to(device)
