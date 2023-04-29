import csv
import pandas as pd
import torch
from torch.utils.data import Dataset
from PreProcess import PreProcess
from transformers import BertTokenizer


class SarcasmData(Dataset):
    def __init__(self, filename):
        self.df = pd.read_csv(filename)
        self.df = self.df.rename(columns={'tweet': 'text'}) if 'tweet' in self.df.columns else self.df
        self.df = self.df.dropna(subset=['text'])  # drop empty rows
        self.sentences = self.df['text'].tolist()  # sarcastic sentences
        self.labels = self.df['sarcastic'].tolist()  # 1 is sarcastic and 0 is non-sarcastic

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        text = self.sentences[idx]
        label = self.labels[idx]
        # Tokenize the text and convert it into numerical inputs
        tokenized_text = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = tokenized_text['input_ids']
        attention_mask = tokenized_text['attention_mask']
        return (input_ids.squeeze(0), attention_mask.squeeze(0), torch.tensor(label))

    def clean_text(self):
        preprocess = PreProcess()
        self.sentences = preprocess.clean_dataset(self.sentences, self.__len__())

