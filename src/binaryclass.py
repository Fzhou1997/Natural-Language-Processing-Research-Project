import numpy as np
import pandas as pd

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torcheval.metrics.functional import binary_f1_score, binary_auroc, binary_accuracy

from transformers import AutoTokenizer, AutoModel

MODEL = 'google-bert/bert-base-uncased'
MAX_LENGTH = 256
BATCH_SIZE = 64

# if you run this file directly inside the src/ directory
# (as opposed to the outer root directory of the project)
# the filepaths should be '../res/{file}.csv' instead
FILEPATH_TRAIN = 'res/train.csv'
FILEPATH_TEST = 'res/test.csv'


def load_dataframe(filepath):
    df = pd.read_csv(filepath)

    # data cleaning
    df = df.drop(['uniqueID', 'date'], axis=1)
    df.columns = ['drug_name', 'condition', 'review', 'rating', 'useful_count']
    df.drug_name = df.drug_name.str.lower()
    df.condition = df.condition.str.lower()

    # remove leading/trailing quotes
    df.review = df.review.map(lambda review: review[1:-1])

    return df


class ReviewDataset(Dataset):
    def __init__(self, filepath, tokenizer):
        self.data = pd.read_csv(filepath).reset_index(drop=True)
        self.tokenizer = tokenizer

        # data cleaning
        self.data = self.data.drop(['uniqueID', 'date'], axis=1)
        self.data.columns = ['drug_name', 'condition',
                             'review', 'rating', 'useful_count']

        # create binary sentiment label
        self.data = self.data[(self.data.rating == 1) | (self.data.rating == 10)].reset_index(drop=True)
        self.data['sentiment'] = np.where(self.data.rating == 1, 0, 1)

        self.data.drug_name = self.data.drug_name.str.lower()
        self.data.condition = self.data.condition.str.lower()

        # remove leading/trailing quotes
        self.data.review = self.data.review.map(lambda review: review[1:-1])

        tokens = tokenizer(
            list(self.data.review.values),
            return_tensors='pt',
            padding='max_length',
            max_length=MAX_LENGTH,
            truncation=True)
        input_ids = torch.squeeze(tokens['input_ids'])
        attention_mask = torch.squeeze(tokens['attention_mask'])
        self.X = dict(input_ids=input_ids, attention_mask=attention_mask)
        self.y = torch.tensor(self.data.sentiment).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return dict(input_ids=self.X['input_ids'][idx], attention_mask=self.X['attention_mask'][idx]), self.y[idx]


class RatingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL, output_attentions=False, return_dict=False)
        for param in self.bert.parameters():
            param.requires_grad = False  # freeze BERT weights

        # linear layer on top to condense features into regressor
        self.linear1 = nn.Linear(768, 512)  # 393728
        self.linear2 = nn.Linear(512, 256)  # 131328
        self.linear3 = nn.Linear(256, 64)   #  16448
        self.linear4 = nn.Linear(64, 2)     #    130

    def forward(self, x):
        x = self.bert(**x)
        x = x[0][:, 0, :]  # only look at [CLS] token
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x


class ModelTrainer:
    def __init__(self, model, loss_fn, optimizer, train_dataset, test_dataset, device, tb_writer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
        self.test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        self.device = device
        self.tb_writer = tb_writer

    def evaluate(self):
        self.model.eval()
        self.model.to(self.device)
        y_test = self.test_dataset.y

        tensors = []
        for (X, _) in tqdm(self.test_dataloader, desc='evaluating on test set'):
            X['input_ids'] = X['input_ids'].to(self.device)
            X['attention_mask'] = X['attention_mask'].to(self.device)
            tensors.append(model(X).detach().cpu())
            X['input_ids'] = X['input_ids'].cpu()
            X['attention_mask'] = X['attention_mask'].cpu()
        y_pred = torch.cat(tensors)

        y_pred = torch.argmax(y_pred, dim=1)

        acc = binary_accuracy(y_pred, y_test)
        print('acc', acc)

        f1 = binary_f1_score(y_pred, y_test)
        print('f1', f1)

        auc = binary_auroc(y_pred, y_test)
        print('auc', auc)

    def train(self):
        self.model.train()
        epoch_number = 0

        EPOCHS = 10
        best_vloss = 1_000_000.
        for epoch in range(EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.train_one_epoch(epoch_number)

            running_vloss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.model.eval()

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in enumerate(self.test_dataloader):
                    vinputs, vlabels = vdata
                    vinputs['input_ids'] = vinputs['input_ids'].to(device)
                    vinputs['attention_mask'] = vinputs['attention_mask'].to(device)
                    vlabels = vlabels.to(device)

                    voutputs = self.model(vinputs)
                    vloss = self.loss_fn(voutputs, vlabels)
                    running_vloss += vloss

                    vinputs['input_ids'] = vinputs['input_ids'].cpu()
                    vinputs['attention_mask'] = vinputs['attention_mask'].cpu()

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            # Log the running loss averaged per batch
            # for both training and validation
            self.tb_writer.add_scalars('Training vs. Validation Loss',
                                       {'Training': avg_loss, 'Validation': avg_vloss},
                                       epoch_number + 1)
            self.tb_writer.flush()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = 'binaryclass_model_{}_{}'.format(timestamp, epoch_number)
                torch.save(self.model.state_dict(), model_path)

            epoch_number += 1

    def train_one_epoch(self, epoch_index):
        running_loss = 0.
        last_loss = 0.

        for idx, (X, y) in enumerate(self.train_dataloader):
            X['input_ids'] = X['input_ids'].to(self.device)
            X['attention_mask'] = X['attention_mask'].to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            y_pred = model(X)
            loss = self.loss_fn(y_pred, y)
            loss.backward()
            self.optimizer.step()

            X['input_ids'] = X['input_ids'].cpu()
            X['attention_mask'] = X['attention_mask'].cpu()

            # log metrics
            running_loss += loss.item()
            if idx % 100 == 99:
                last_loss = running_loss / 100  # loss per batch
                print('  batch {} loss: {}'.format(idx + 1, last_loss))
                tb_x = epoch_index * len(self.train_dataloader) + idx + 1
                self.tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss


if __name__ == '__main__':
    device = (
        'cuda'
        if torch.cuda.is_available()
        else 'mps'
        if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'Using {device} device')

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    train_dataset = ReviewDataset(FILEPATH_TRAIN, tokenizer)
    print(f'{len(train_dataset)} training samples loaded')

    test_dataset = ReviewDataset(FILEPATH_TEST, tokenizer)
    print(f'{len(test_dataset)} testing samples loaded')

    model = RatingModel().to(device)
    # 11 # load previous version tgo continue training
    model.load_state_dict(torch.load('binaryclass_model_20240414_103223_8'))
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters())

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/review_trainer_{}'.format(timestamp))

    trainer = ModelTrainer(
        model,
        loss_fn,
        optimizer,
        train_dataset,
        test_dataset,
        device,
        writer
    )

    # trainer.train()
    trainer.evaluate()
