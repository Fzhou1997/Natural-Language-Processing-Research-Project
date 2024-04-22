import numpy.typing as npt
import torch
from lime.lime_text import LimeTextExplainer

from src.classifier_binary import Classifier
from src.tokenizer_bert import Tokenizer


class Explainer:
    def __init__(self, classifier: Classifier, tokenizer: Tokenizer):
        classifier.eval()
        self.classifier = classifier
        self.tokenizer = tokenizer

        def predict(x: list[str]) -> npt.Array:
            tokenized = self.tokenizer(x)
            predicted = self.classifier(tokenized)
            return predicted.cpu().numpy()

        self.predictor = predict
        self.explainer = LimeTextExplainer(class_names=['negative', 'positive'])

    def __call__(self, x: list[str]):
        self.classifier.eval()
        explanation = self.explainer.explain_instance(x, self.predictor, num_samples=)

        pass
