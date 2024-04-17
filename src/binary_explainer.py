from os import PathLike
from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd
import shap
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.binaryclass import RatingModel

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

FILEPATH_TRAIN = '../../res/train.csv'
FILEPATH_TEST = '../../res/test.csv'
FILEPATH_ALL = '../res/all.csv'
MAX_LENGTH = 256


class Tokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')

    def __call__(self, reviews: str | list[str] | npt.NDArray | dict) -> dict[str, torch.Tensor]:
        if isinstance(reviews, np.ndarray):
            reviews = reviews.tolist()
        elif isinstance(reviews, str):
            reviews = [reviews]
        elif isinstance(reviews, dict):
            reviews = reviews['text']
        tokens = self.tokenizer(
            reviews,
            return_tensors='pt',
            padding='max_length',
            max_length=MAX_LENGTH,
            truncation=True)
        input_ids = torch.squeeze(tokens['input_ids']).to(device)
        attention_mask = torch.squeeze(tokens['attention_mask']).to(device)
        if len(reviews) == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
        return dict(input_ids=input_ids, attention_mask=attention_mask)

    def get_tokenizer(self):
        return self.tokenizer

    def get_mask_token_id(self):
        return self.tokenizer.mask_token_id


class Classifier:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.classifier = RatingModel()
        self.classifier.to(device)
        self.classifier.load_state_dict(torch.load('../models/binaryclass_model_20240414_103223_8'))

    def __call__(self, x: str | list[str] | npt.NDArray):
        return self.classifier(self.tokenizer(x))


class Explainer:
    def __init__(self, classifier: Classifier, tokenizer: Tokenizer):
        self.explainer = shap.Explainer(classifier, tokenizer.get_tokenizer())

    def __call__(self, x: list[str]):
        explanation = self.explainer(x, fixed_context=1)
        return Explanation(explanation)


class Explanation:
    def __init__(self, explanation: shap.Explanation):
        self.explanation = explanation
        self.contributions = {}

        data = explanation.data
        values = explanation.values

        for i, doc in enumerate(data):
            for j, token in enumerate(doc):
                if token not in self.contributions:
                    self.contributions[token] = {}
                self.contributions[token]["count"] = self.contributions[token].get("count", 0) + 1
                self.contributions[token]["positive sum"] = self.contributions[token].get("positive sum", 0) + values[i][j][1]
                self.contributions[token]["negative sum"] = self.contributions[token].get("negative sum", 0) + values[i][j][0]

        for token in self.contributions:
            self.contributions[token]["positive mean"] = self.contributions[token]["positive sum"] / self.contributions[token]["count"]
            self.contributions[token]["negative mean"] = self.contributions[token]["negative sum"] / self.contributions[token]["count"]

    def get_contributions(self):
        return self.contributions

    def get_top_positive_tokens(self, n: int):
        return sorted(self.contributions, key=lambda x: self.contributions[x]["positive mean"], reverse=True)[:n]

    def get_top_negative_tokens(self, n: int):
        return sorted(self.contributions, key=lambda x: self.contributions[x]["negative mean"], reverse=True)[:n]


class RawReviewDataset(Dataset):
    def __init__(self, filepath: str | bytes | PathLike[str] | PathLike[bytes] = None):
        if filepath is None:
            self.data = None
        else:
            self.data = pd.read_csv(filepath).reset_index(drop=True)
            self.data = self.data.drop(['uniqueID', 'date', 'usefulCount'], axis=1)
            self.data.columns = ['drug_name', 'condition', 'review', 'rating']
            self.data['sentiment'] = np.where(self.data.rating == 1, 0, 1)
            self.data = self.data.drop(['rating'], axis=1)
            self.data.drug_name = self.data.drug_name.str.lower()
            self.data.condition = self.data.condition.str.lower()
            self.data.review = self.data.review.map(lambda review: review[1:-1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rows = self.data.iloc[idx]
        return {"text": list(rows.review), "label": list(rows.sentiment)}

    def _set_data(self, data: pd.DataFrame):
        self.data = data
        return self

    def get_drug_names(self) -> list[str]:
        return self.data.drug_name.unique().tolist()

    def get_conditions(self) -> list[str]:
        return self.data.condition.unique().tolist()

    def get_subset_drug_name(self, drug_name: str) -> Self:
        return RawReviewDataset()._set_data(self.data[self.data.drug_name == drug_name].reset_index(drop=True))

    def get_subset_condition(self, condition: str) -> Self:
        return RawReviewDataset()._set_data(self.data[self.data.condition == condition].reset_index(drop=True))

    def get_subset_drug_name_condition(self, drug_name: str, condition: str) -> Self:
        return RawReviewDataset()._set_data(self.data[(self.data.drug_name == drug_name) & (self.data.condition == condition)].reset_index(drop=True))


if __name__ == '__main__':
    tokenizer = Tokenizer()
    classifier = Classifier(tokenizer)
    explainer = Explainer(classifier, tokenizer)
    dataset = RawReviewDataset(FILEPATH_ALL)
    subset = dataset.get_subset_drug_name(dataset.get_drug_names()[0])
    explanation = explainer(subset[:10])
    print("Top 32 tokens correlated with positive model output:", explanation.get_top_positive_tokens(32))
    print("Top 32 tokens correlated with negative model output:", explanation.get_top_negative_tokens(32))
