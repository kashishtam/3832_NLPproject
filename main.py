from DataLoader import SarcasmData
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

# Load and clean dataset
dataset = SarcasmData("train/train.EN.csv")
dataset.clean_text()

# returns tensors of (tokenized_ids, attention_mask, sarcasm level)
print(dataset[1])

# creating Data loader
dataLoader = DataLoader(dataset, batch_size=32, shuffle=True)

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
# Train the model
for epoch in range(num_epochs):
    print('### Epoch: ' + str(epoch + 1) + ' ###')
    running_loss = 0.0
    for step, batch in enumerate(dataLoader):
        # Unpack the inputs and labels
        inputs, attention_mask, labels = batch

        # move tensors to GPU take advantage of the parallel computing power of a GPU to speed up the training process.
        inputs, attention_mask, labels = inputs.to(device), attention_mask.to(device), labels.to(device)

        # Clear the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs, attention_mask=attention_mask)[0]

        # Compute the loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update the parameters
        optimizer.step()

        # Update the running loss
        running_loss += loss.item()

        if step % print_frequency == (print_frequency - 1):
            print(f'Epoch: {epoch + 1}, Batch: {step}, Loss: {running_loss}')

    # Print the epoch loss
    epoch_loss = running_loss / len(dataset)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')
