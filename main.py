from DataLoader import SarcasmData
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn as nn
import pandas as pd
import training
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Split training and validation data
df = pd.read_csv('train/train.En.csv')
train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)
# Save the datasets as CSV files
train_data.to_csv('train/train_data.csv', index=False)
val_data.to_csv('train/val_data.csv', index=False)

# Load and clean train and valid dataset
train_dataset = SarcasmData("train/train_data.csv")
valid_dataset = SarcasmData('train/val_data.csv')

# DataLoader
train_dataLoader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_dataLoader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

# Load and clean test dataset
test_dataset = SarcasmData("test/task_A_En_test.csv")
test_dataset.clean_text()
test_dataLoader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Set up the optimizer and the loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train model
num_epochs = 3
training.train_model(model, num_epochs, train_dataLoader, valid_dataLoader, criterion, optimizer, device)

# Evaluate model on test set
all_preds, f1 = training.evaluate_model(model, test_dataLoader, device)
print(f'Test F1 Score: {f1:.4f}')

# output file
df = pd.read_csv('test/task_A_En_test.csv')
df['sarcasm'] = all_preds
df.to_csv('output.csv', index=False)
