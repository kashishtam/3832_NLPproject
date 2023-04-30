from DataLoader import SarcasmData
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn as nn
import pandas as pd
import training
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os


# Leave as '' to generate new model, otherwise fill in with name of model file to use
model_to_load = ''

# Parameters
num_batches = 32
num_epochs = 3
learning_rate = 2e-5

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
train_dataLoader = DataLoader(train_dataset, batch_size=num_batches, shuffle=True)
valid_dataLoader = DataLoader(valid_dataset, batch_size=num_batches, shuffle=False)

# Load and clean test dataset
test_dataset = SarcasmData("test/task_A_En_test.csv")
test_dataset.clean_text()
test_dataLoader = DataLoader(test_dataset, batch_size=num_batches, shuffle=False)

# model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Set up the optimizer and the loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train model
if model_to_load == '':
    training.train_model(model, num_epochs, train_dataLoader, valid_dataLoader, criterion, optimizer, device)
else:
    model.load_state_dict(torch.load(model_to_load))
    model.eval()

# Evaluate model on test set
all_preds, f1 = training.evaluate_model(model, test_dataLoader, device)
print(f'Test F1 Score: {f1:.4f}')

# Output file
df = pd.read_csv('test/task_A_En_test.csv')
df['sarcastic'] = all_preds
df.to_csv('output.csv', index=False)

# If current model is better than the best model, overwrite best model with it
f = open('best_f1.txt', 'r')
best_score = float(f.read())
print(best_score)
if(f1 > best_score):
    print('\nPrevious best model had an f1 score of ' + str(f1) + '. Overwriting best model.')
    if os.path.isfile("best_model.pt"):
        os.remove("best_model.pt")
    torch.save(model.state_dict, "best_model.pt")
    f.close()
    f = open('best_f1.txt', 'w')
    f.seek(0)
    f.truncate()
    f.write(str(f1))
# Optional save model
inp = input('\nSave model to name? Y/N\n')
if inp == 'Y':
    inp = input('Save as:\n')
    torch.save(model.state_dict(), inp + ".pt")
