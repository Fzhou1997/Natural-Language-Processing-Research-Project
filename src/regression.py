# import numpy as np
import pandas as pd

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel


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
        self.data.drug_name = self.data.drug_name.str.lower()
        self.data.condition = self.data.condition.str.lower()

        # remove leading/trailing quotes
        self.data.review = self.data.review.map(lambda review: review[1:-1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        review = self.data.review[idx]
        tokens = tokenizer(
            review,
            return_tensors='pt',
            padding='max_length',
            max_length=MAX_LENGTH,
            truncation=True)
        input_ids = torch.squeeze(tokens['input_ids'])
        attention_mask = torch.squeeze(tokens['attention_mask'])

        rating = torch.tensor(self.data.rating[idx]).float()
        return dict(input_ids=input_ids, attention_mask=attention_mask), rating


class RatingModel(nn.Module):
    def __init__(self, bert_hidden_size=768, hidden_size=256):
        super().__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased',
                                              hidden_size=bert_hidden_size,
                                              output_attentions=False,
                                              return_dict=False)
        for param in self.bert.parameters():
            param.requires_grad = False  # freeze BERT weights

        # linear layer on top to condense features into regressor
        self.linear1 = nn.Linear(bert_hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.bert(**x)
        x = x[0][:, 0, :]  # only look at [CLS] token
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class ModelTrainer:
    def __init__(self, model, loss_fn, optimizer, train_dataloader, test_dataloader, device):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device

    def train_one_epoch(self, epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.

        for idx, (X, y) in enumerate(self.train_dataloader):
            X['input_ids'] = X['input_ids'].to(self.device)
            X['attention_mask'] = X['attention_mask'].to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            y_pred = model(X)
            loss = self.loss_fn(y_pred.squeeze(), y)
            loss.backward()
            self.optimizer.step()

            # log metrics
            running_loss += loss.item()
            if idx % BATCH_SIZE == BATCH_SIZE - 1:
                last_loss = running_loss / BATCH_SIZE  # loss per batch
                print('  batch {} loss: {}'.format(idx + 1, last_loss))
                tb_x = epoch_index * len(self.train_dataloader) + idx + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
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

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = ReviewDataset(FILEPATH_TRAIN, tokenizer)
    print(f'{len(train_dataset)} training samples loaded')

    test_dataset = ReviewDataset(FILEPATH_TEST, tokenizer)
    print(f'{len(test_dataset)} testing samples loaded')

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = RatingModel().to(device)
    loss_fn = torch.nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters())

    trainer = ModelTrainer(
        model,
        loss_fn,
        optimizer,
        train_dataloader,
        test_dataloader,
        device
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/review_trainer_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 5
    best_vloss = 1_000_000.
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = trainer.train_one_epoch(epoch_number, writer)

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(test_dataloader):
                vinputs, vlabels = vdata
                vinputs['input_ids'] = vinputs['input_ids'].to(device)
                vinputs['attention_mask'] = vinputs['attention_mask'].to(device)
                vlabels = vlabels.to(device)


                voutputs = model(vinputs)
                vloss = loss_fn(voutputs.squeeze(), vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1
