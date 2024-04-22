import json
import re
from os import PathLike
from typing import Self

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import shap
import torch

from src.classifier_binary import Classifier
from src.tokenizer_bert import Tokenizer

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Explanation:
    def __init__(self, explanation: shap.Explanation = None):
        self.contributions = {}
        if explanation is None:
            return
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

    def get_contributions(self) -> dict[str, dict[str, int | float]]:
        return self.contributions

    def get_top_positive_tokens(self, n: int) -> list[str]:
        return sorted(self.contributions, key=lambda x: self.contributions[x]["positive mean"], reverse=True)[:n]

    def get_top_positive_tuples(self, n: int) -> list[tuple[str, float]]:
        top_positive_tokens = self.get_top_positive_tokens(n)
        return [(token,  self.contributions[token]['positive mean']) for token in top_positive_tokens]

    def get_top_negative_tokens(self, n: int) -> list[str]:
        return sorted(self.contributions, key=lambda x: self.contributions[x]["negative mean"], reverse=True)[:n]

    def get_top_negative_tuples(self, n: int) -> list[tuple[str, float]]:
        top_negative_tokens = self.get_top_negative_tokens(n)
        return [(token,  self.contributions[token]['negative mean']) for token in top_negative_tokens]

    def save(self, path: str | bytes | PathLike[str] | PathLike[bytes]) -> None:
        with open(path, "w") as f:
            json.dump(self.contributions, f)

    def load(self, path: str | bytes | PathLike[str] | PathLike[bytes]) -> Self:
        with open(path, "r") as f:
            self.contributions = json.load(f)
        return self

    def union(self, other: Self) -> Self:
        for token in other.contributions:
            if token not in self.contributions:
                self.contributions[token] = other.contributions[token]
            else:
                self.contributions[token]["count"] += other.contributions[token]["count"]
                self.contributions[token]["positive sum"] += other.contributions[token]["positive sum"]
                self.contributions[token]["negative sum"] += other.contributions[token]["negative sum"]
                self.contributions[token]["positive mean"] = self.contributions[token]["positive sum"] / self.contributions[token]["count"]
                self.contributions[token]["negative mean"] = self.contributions[token]["negative sum"] / self.contributions[token]["count"]
        return self

    def plot_top_tokens(self, n: int) -> None:
        top_positive_tokens = self.get_top_positive_tokens(n)
        top_negative_tokens = self.get_top_negative_tokens(n)[::-1]

        top_tokens = top_positive_tokens + top_negative_tokens
        top_means = [self.contributions[token]["positive mean"] for token in top_positive_tokens] + [-self.contributions[token]["negative mean"] for token in top_negative_tokens]
        top_labels = ["positive"] * n + ["negative"] * n

        df = pd.DataFrame({"token": top_tokens, "mean": top_means, "label": top_labels})
        plt.figure(figsize=(8, 6))
        custom_palette = sns.color_palette(["#0080ff", "#ff0000"])
        sns.barplot(data=df, x="mean", y="token", hue="label", dodge=False, palette=custom_palette)
        plt.title(f"Top {n} positive and negative token contributions")
        plt.show()


class Explainer:
    def __init__(self, classifier: Classifier, tokenizer: Tokenizer):
        classifier.eval()
        self.classifier = classifier.to(device)
        self.tokenizer = tokenizer

        def classify(x: str | list[str]) -> torch.Tensor:
            tokens = self.tokenizer(x)
            with torch.no_grad():
                y = self.classifier(tokens)
            return y

        self.explainer = shap.Explainer(classify, self.tokenizer.get_tokenizer())

    def __call__(self, x: list[str]) -> Explanation:
        self.classifier.eval()
        explanation = self.explainer(x, fixed_context=1)
        return Explanation(explanation)
