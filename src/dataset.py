from os import PathLike
from typing import Self

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class ReviewDataset(Dataset):
    def __init__(self, filepath: str | bytes | PathLike[str] | PathLike[bytes] = None, preprocess: bool = True):
        if filepath is None:
            self.data = None
        else:
            self.data = pd.read_csv(filepath).reset_index(drop=True)
            self.data = self.data.drop(['uniqueID', 'date', 'usefulCount'], axis=1)
            self.data.columns = ['drug_name', 'condition', 'review', 'rating']
            self.data.drug_name = self.data.drug_name.str.lower()
            self.data.condition = self.data.condition.str.lower()
            self.data.review = self.data.review.map(lambda review: review[1:-1])
            if preprocess:
                self.encode_binary_sentiment()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int | list[int]) -> tuple[list[str], list[int]]:
        rows = self.data.iloc[idx]
        features = rows.review
        labels = rows.sentiment if 'sentiment' in self.data.columns else rows.rating
        if isinstance(idx, int):
            return [features], [labels]
        else:
            return features.tolist(), labels.tolist()

    def _set_data(self, data: pd.DataFrame) -> Self:
        self.data = data
        return self

    def encode_binary_sentiment(self) -> Self:
        self.data = self.data[(self.data.rating == 1) | (self.data.rating == 10)].reset_index(drop=True)
        self.data['sentiment'] = np.where(self.data.rating == 1, 0, 1)
        return self

    def get_drug_names(self) -> list[str]:
        return self.data.drug_name.unique().tolist()

    def get_conditions(self) -> list[str]:
        return self.data.condition.unique().tolist()

    def get_class_counts(self) -> dict[str, int]:
        if 'sentiment' in self.data.columns:
            return self.data.sentiment.value_counts().sort_index()
        else:
            return self.data.rating.value_counts().sort_index()

    def get_subset_drug_name(self, drug_name: str) -> Self:
        return ReviewDataset()._set_data(self.data[self.data.drug_name == drug_name].reset_index(drop=True))

    def get_subset_condition(self, condition: str) -> Self:
        return ReviewDataset()._set_data(self.data[self.data.condition == condition].reset_index(drop=True))

    def get_subset_drug_name_condition(self, drug_name: str, condition: str) -> Self:
        return ReviewDataset()._set_data(self.data[(self.data.drug_name == drug_name) & (self.data.condition == condition)].reset_index(drop=True))

    def head(self, n: int = 10) -> pd.DataFrame:
        return self.data.head(n)
