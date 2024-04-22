from os import PathLike

import torch
from sklearn.metrics import confusion_matrix
from torch.nn import Module, Linear
from torch.nn.functional import relu, softmax
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy, F1Score, AUROC
from transformers import AutoModel

from src.dataloader_tokenized import ReviewDataLoader

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Classifier(Module):
    def __init__(self, model: str):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model, output_attentions=False, return_dict=False)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.linear1 = Linear(768, 512)
        self.linear2 = Linear(512, 256)
        self.linear3 = Linear(256, 64)
        self.linear4 = Linear(64, 2)

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        y = self.bert(**x)
        y = y[0][:, 0, :]
        y = relu(self.linear1(y))
        y = relu(self.linear2(y))
        y = relu(self.linear3(y))
        y = self.linear4(y)
        return y


class Trainer:
    def __init__(self,
                 model: Classifier,
                 loss_fn: _Loss,
                 optimizer: Optimizer,
                 train_loader: ReviewDataLoader,
                 test_loader: ReviewDataLoader,
                 writer: SummaryWriter,
                 out_path: str | bytes | PathLike[str] | PathLike[bytes]):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.accuracy_fn = Accuracy(task='binary').to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.writer = writer
        self.out_path = out_path

    def _train_one_epoch(self) -> tuple[float, float]:
        running_loss = 0
        self.accuracy_fn.reset()
        self.model.train()
        for features, labels in self.train_loader:
            self.optimizer.zero_grad()
            predicted = self.model(features)
            loss = self.loss_fn(predicted, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            probs = softmax(predicted, dim=1)
            pred = probs.argmax(dim=1)
            self.accuracy_fn.update(pred, labels)
        running_loss /= len(self.train_loader)
        accuracy = self.accuracy_fn.compute().cpu().item()
        return running_loss, accuracy

    def _validate_one_epoch(self) -> tuple[float, float]:
        loss = 0
        self.accuracy_fn.reset()
        self.model.eval()
        with torch.no_grad():
            for features, labels in self.test_loader:
                predicted = self.model(features)
                loss += self.loss_fn(predicted, labels).item()
                probs = softmax(predicted, dim=1)
                pred = probs.argmax(dim=1)
                self.accuracy_fn.update(pred, labels)
        loss /= len(self.test_loader)
        accuracy = self.accuracy_fn.compute().cpu().item()
        return loss, accuracy

    def train(self, epochs: int) -> tuple[list[float], list[float], list[float], list[float]]:
        best_loss = float('inf')
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        for epoch in range(epochs):
            train_loss, train_accuracy = self._train_one_epoch()
            val_loss, val_accuracy = self._validate_one_epoch()
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_accuracy, epoch)
            self.writer.add_scalar('Accuracy/val', val_accuracy, epoch)
            self.writer.flush()
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            print(f'Epoch {epoch + 1}/{epochs}')
            print(f'Train Loss: {train_loss:.6f} | Train Accuracy: {train_accuracy:.6f}')
            print(f'Val Loss: {val_loss:.6f} | Val Accuracy: {val_accuracy:.6f}')
            if val_loss < best_loss:
                print(f'Val loss decreased from {best_loss:.6f} to {val_loss:.6f}. Saving model...')
                best_loss = val_loss
                torch.save(self.model.state_dict(), self.out_path)
            print(25*"==")
        return train_losses, train_accuracies, val_losses, val_accuracies


class Tester:
    def __init__(self,
                 model: Classifier,
                 loss_fn: _Loss,
                 test_loader: ReviewDataLoader):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.accuracy_fn = Accuracy(task='binary').to(device)
        self.f1_score_fn = F1Score(task='binary').to(device)
        self.auroc_fn = AUROC(task='binary').to(device)
        self.test_loader = test_loader
        self.confusion_matrix_fn = confusion_matrix

    def test(self) -> tuple[float, float, float, float, list[list[int]]]:
        self.model.eval()
        loss = 0
        self.accuracy_fn.reset()
        self.f1_score_fn.reset()
        self.auroc_fn.reset()
        test_labels = torch.empty(0, dtype=torch.long, device=device)
        test_predictions = torch.empty(0, dtype=torch.long, device=device)
        with torch.no_grad():
            for features, labels in self.test_loader:
                predicted = self.model(features)
                loss += self.loss_fn(predicted, labels).item()
                probs = softmax(predicted, dim=1)
                pred = probs.argmax(dim=1)
                self.accuracy_fn.update(pred, labels)
                self.f1_score_fn.update(pred, labels)
                self.auroc_fn.update(pred, labels)
                test_labels = torch.cat((test_labels, labels))
                test_predictions = torch.cat((test_predictions, pred))
        loss /= len(self.test_loader)
        accuracy = self.accuracy_fn.compute().cpu()
        f1_score = self.f1_score_fn.compute().cpu()
        auroc = self.auroc_fn.compute().cpu()
        test_labels = test_labels.cpu().numpy()
        test_predictions = test_predictions.cpu().numpy()
        cm = self.confusion_matrix_fn(test_labels, test_predictions)
        return loss, accuracy, f1_score, auroc, cm
