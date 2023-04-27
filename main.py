import os, random, sys, matplotlib.pyplot as plt
from DataLoader import SarcasmData, test_SarcasmData
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn as nn
import utils
import training

from torch.utils.data import DataLoader


# Load and clean training dataset
dataset = SarcasmData("train/train.EN.csv")
dataset.clean_text()

# Load and clean test dataset
test_dataset = test_SarcasmData("test/task_A_En_test.csv")
test_dataset.clean_text()

# returns tensors of (tokenized_ids, attention_mask, sarcasm level)
print(dataset[1])
print(test_dataset[1])

# softmax function
softmax = nn.Softmax(dim=1)

# creating Data loader
dataLoader = DataLoader(dataset, batch_size=16, shuffle=True)

# Load pretrained model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Set up the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# Set the device to use for training (e.g., GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print_frequency = 10
num_epochs = 3

# Not yet implemented(may not need)
running_loss_list = []

training.train_model(model, num_epochs, dataLoader, criterion, optimizer, device)

# Train the model
# for epoch in range(num_epochs):
#     print('### Epoch: ' + str(epoch + 1) + ' ###')

#     running_loss = 0.0

#     model.train()

#     for step, batch in enumerate(dataLoader):
#         # Unpack the inputs and labels
#         inputs, attention_mask, labels = batch

#         # move tensors to GPU take advantage of the parallel computing power of a GPU to speed up the training process.
#         inputs, attention_mask, labels = inputs.to(device), attention_mask.to(device), labels.to(device)

#         # Clear the gradients
#         optimizer.zero_grad()

#         # Forward pass
#         outputs = model(inputs, attention_mask=attention_mask)[0]
#         output_probs = softmax(outputs)

#         # Compute the loss
#         loss = criterion(output_probs.squeeze(1), labels)

#         # Backward pass
#         loss.backward()

#         # Update the parameters
#         optimizer.step()

#         # Update the running loss
#         running_loss += loss.item()

#         if step % print_frequency == 1:
#             print(f'Epoch: {epoch + 1}, Batch: {step}, Loss: {running_loss/print_frequency}')
#             running_loss_list.append(running_loss)
#             running_loss = 0
