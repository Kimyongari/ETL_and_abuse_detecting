from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import Trainer, TrainingArguments
from tqdm import tqdm
import pandas as pd
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from tqdm import tqdm
from datetime import datetime
import os
import json

class Model(nn.Module):
    def __init__(self, emb_model_name='klue/bert-base'):
        super(Model, self).__init__()
        self.emb = AutoModel.from_pretrained(emb_model_name).get_input_embeddings()
        self._1DCNN_1 = nn.Conv1d(768, 128, kernel_size=5, padding=2)
        self._1DCNN_2 = nn.Conv1d(128, 32, kernel_size=3, padding=1)
        self._FC_1 = nn.Linear(32*16, 128)
        self._FC_2 = nn.Linear(128, 1)
        self._act = nn.LeakyReLU()
        self._pool = nn.MaxPool1d(2)
        self._dropout = nn.Dropout(0.4)
        self.sigmoid = nn.Sigmoid()
        self._loss = nn.BCELoss()
    
    def forward(self, input_ids,):

        x = self.emb(input_ids) # B, L, D
        x = x.permute(0, 2, 1) # B, D, L
        x = self._1DCNN_1(x)
        x = self._act(x)
        x = self._pool(x)
        x = self._1DCNN_2(x)
        x = self._act(x)
        x = self._pool(x)
        x = x.view(x.size(0), -1)
        x = self._FC_1(x)
        x = self._act(x)
        x = self._dropout(x)
        x = self._FC_2(x)
        logits = self.sigmoid(x)
        return logits

class CustomDataset(Dataset):
    def __init__(self, texts, labels,emb_model_name, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(emb_model_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
        }
def make_dataloader(df, emb_model_name, batch_size = 32):

    def custom_collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
        labels = torch.stack(labels)  
        
        return {'input_ids': input_ids, 'labels': labels}
    
    dataset_train, dataset_valid = train_test_split(df, test_size=0.3, random_state=42)

    dataset_train = dataset_train.reset_index(drop=True)
    dataset_valid = dataset_valid.reset_index(drop=True)

    train_dataset = CustomDataset(df['text'], df['label'], emb_model_name)
    valid_dataset = CustomDataset(df['text'], df['label'], emb_model_name)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=True, collate_fn=custom_collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                                shuffle=True, collate_fn=custom_collate_fn)

    return train_dataloader, valid_dataloader

def total_metric(logits, labels):
    preds = (logits > 0.5).float()
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    auc = roc_auc_score(labels, preds)
    return {'accuracy': round(acc,4), 'f1': round(f1,4), 'roc_auc': round(auc,4)}

def train(model, df, emb_model_name, EPOCH=5, batch_size=32):
    train_dataloader, valid_dataloader = make_dataloader(df, emb_model_name, batch_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr = 2e-4)
    os.makedirs('results', exist_ok=True)
    if os.path.isfile('results/best_model_metrics.json'):
        with open('results/best_model_metrics.json', 'r') as f:
            best_metrics = json.load(f)
            best_acc = best_metrics['accuracy']
            best_f1 = best_metrics['f1']
            best_roc_auc = best_metrics['roc_auc']
    else:
        best_acc = 0
        best_f1 = 0
        best_roc_auc = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'train on {device}')
    model = model.to(device)
    for epoch in range(EPOCH):
        model.train()
        for batch in tqdm(train_dataloader, desc=f'epoch: {epoch+1}'):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids)

            loss = model._loss(logits, labels.view(-1,1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        with torch.no_grad():
            model.eval()
            all_logits = []
            all_labels = []
            all_eval_loss = []
            for batch in valid_dataloader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                logits = model(input_ids)
                eval_loss = model._loss(logits, labels.view(-1,1))
                all_logits.append(logits)
                all_labels.append(labels)
                all_eval_loss.append(eval_loss)

            metric = total_metric(torch.cat(all_logits), torch.cat(all_labels))
            if metric['accuracy'] > best_acc and metric['f1'] > best_f1 and metric['roc_auc'] > best_roc_auc:  
                best_acc = metric['accuracy']
                best_f1 = metric['f1']
                with open('results/best_model_metrics.json', 'w') as f:
                    json.dump({
                        'accuracy': metric['accuracy'],
                        'f1': metric['f1'],
                        'roc_auc': metric['roc_auc'],
                        'updated' : datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }, f)
                path = f'results/best_model.pth'
                torch.save(model.state_dict(), path)
                print('saved best model.')
            eval_loss = torch.stack(all_eval_loss).mean()
        print(f'loss: {round(loss.item(), 4)}', f'eval_loss: {round(eval_loss.item(), 4)}')
        print(f'accuracy: {metric["accuracy"]}, f1: {metric["f1"]}, roc_auc: {metric["roc_auc"]}')

def predict(texts, model, emb_model_name='klue/bert-base'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'predict on {device}')
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(emb_model_name)
    input_ids = tokenizer(texts, truncation=True, padding="max_length", max_length=64, return_tensors="pt")['input_ids'].to(device)
    logits = model(input_ids)
    return logits.detach()

def load_best_model():
    model = Model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.load_state_dict(torch.load('results/best_model.pth'))
    model.to(device)
    print('loaded best model.')
    return model
    
class _1DCNN_MODEL:
    def __init__(self, args):
        self.model = Model()
        self.args = args

    def load_best_model(self):
        self.model = load_best_model()

    def train(self, df):
        train(self.model, df, self.args.emb_model_name, self.args.EPOCH, self.args.batch_size)
    
    def predict(self, texts,):
        return predict(texts, self.model, self.args.emb_model_name)
