from DataLoader import CSVReader_train
from DataLoader import CSVReader_test
from PreProcess import PreProcess

from transformers import BertTokenizer
from model import SarcasmDetector

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

training_data = CSVReader_train("train/train.EN.csv")
training_data.read_file()

# TODO: Preprocess data through PreProcess class
processed_data = None  # placeholder
# .....

# TODO: tokenize and convert to numerical inputs
# ....

# creating Data loader
dataLoader = DataLoader(processed_data, batch_size=32, shuffle=True)

# parameters
input_dim = training_data.__len__()  # vocabulary size
hidden_dim = 64
output_dim = 2  # binary classification task (sarcasm or not)

# initializing model
model = SarcasmDetector(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# TODO: Training loop by iterating through data in batches
# ....
