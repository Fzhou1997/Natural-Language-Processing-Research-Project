import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
from transformers import AutoTokenizer

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Tokenizer:
    def __init__(self,
                 model: str,
                 max_length: int):
        self.tokenizer = AutoTokenizer.from_pretrained(model, do_lower_case=True, max_length=256, padding="max_length", truncation=True)
        self.max_length = max_length
        self.special_tokens_map = self.tokenizer.special_tokens_map

    def __call__(self, reviews: str | list[str] | NDArray | Tensor | tuple) -> dict[str, torch.Tensor]:
        if isinstance(reviews, str):
            reviews = [reviews]
        elif isinstance(reviews, np.ndarray):
            reviews = reviews.tolist()
        elif isinstance(reviews, tuple):
            reviews = reviews[0]
        tokens = self.tokenizer(
            reviews,
            return_tensors='pt',
            padding='max_length',
            max_length=self.max_length,
            truncation=True)
        input_ids = torch.squeeze(tokens['input_ids']).to(device)
        attention_mask = torch.squeeze(tokens['attention_mask']).to(device)
        if len(reviews) == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
        return dict(input_ids=input_ids, attention_mask=attention_mask)

    def get_tokenizer(self) -> AutoTokenizer:
        return self.tokenizer

    def get_vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def get_mask_token(self) -> str:
        return self.tokenizer.mask_token

    def get_mask_token_id(self) -> int:
        return self.tokenizer.mask_token_id

    def get_pad_token(self) -> str:
        return self.tokenizer.pad_token

    def get_pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    def get_unk_token(self) -> str:
        return self.tokenizer.unk_token

    def get_unk_token_id(self) -> int:
        return self.tokenizer.unk_token_id
