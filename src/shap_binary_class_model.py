import pandas as pd
import shap
import torch
from transformers import AutoTokenizer

from src.binaryclass import RatingModel

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

FILEPATH_TRAIN = 'res/train.csv'
FILEPATH_TEST = 'res/test.csv'
FILEPATH_ALL = '../res/all.csv'
MAX_LENGTH = 256


class RawReviewDataset:
    def __init__(self, filepath):
        self.data = pd.read_csv(filepath).reset_index(drop=True)
        self.data = self.data.drop(['uniqueID', 'rating', 'date', 'usefulCount'], axis=1)
        self.data.columns = ['drug_name', 'condition', 'review']
        self.data.drug_name = self.data.drug_name.str.lower()
        self.data.condition = self.data.condition.str.lower()
        self.data.review = self.data.review.map(lambda review: review[1:-1])

    def __len__(self):
        return len(self.data)

    def get_drug_names(self) -> list[str]:
        return self.data.drug_name.unique().tolist()

    def get_conditions(self) -> list[str]:
        return self.data.condition.unique().tolist()

    def get_subset_drug_name(self, drug_name: str) -> list[str]:
        return self.data[self.data.drug_name == drug_name].review.tolist()

    def get_subset_condition(self, condition: str) -> list[str]:
        return self.data[self.data.condition == condition].review.tolist()

    def get_subset_drug_name_condition(self, drug_name: str, condition: str) -> list[str]:
        return self.data[(self.data.drug_name == drug_name) & (self.data.condition == condition)].review.tolist()


class ModelExplainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        def tokenize(review: str) -> dict[str, torch.Tensor]:
            tokens = self.tokenizer(
                review,
                return_tensors='pt',
                padding='max_length',
                max_length=MAX_LENGTH,
                truncation=True)
            input_ids = torch.squeeze(tokens['input_ids'])
            attention_mask = torch.squeeze(tokens['attention_mask'])
            return dict(input_ids=input_ids, attention_mask=attention_mask)

        self.explainer = shap.Explainer(model, tokenize)

    def explain(self, reviews: list[str], save_path: str = None):
        return self.explainer(reviews)


if __name__ == '__main__':
    base_model = 'google-bert/bert-base-uncased'
    tuned_model = '../models/binaryclass_model_20240414_103223_8'

    binary_class_model = RatingModel().to(device)
    binary_class_model.load_state_dict(torch.load(tuned_model))

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    explainer = ModelExplainer(binary_class_model, tokenizer)

    raw_review_dataset = RawReviewDataset(FILEPATH_ALL)
    drug_names = raw_review_dataset.get_drug_names()
    print(drug_names[0])
    drug_reviews = raw_review_dataset.get_subset_drug_name(drug_names[0])
    print(drug_reviews[0])

    explanation = explainer.explain(drug_reviews)
    print(explanation)
