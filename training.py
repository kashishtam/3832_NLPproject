import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.metrics import f1_score

def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)
    
def resume(model, filename):
    model.load_state_dict(torch.load(filename))


def train_model(model, num_epochs, train_dataloader, val_dataloader, criterion, optimizer, device):
    print_frequency = 10
    softmax = nn.Softmax(dim=1)

    for epoch in range(num_epochs):
        print('### Epoch: ' + str(epoch + 1) + ' ###')

        running_loss = 0.0

        model.train()
        model.to(device)

        f = open('models/best_f1.txt', 'r')
        best_score = float(f.read())
        f.close()

        for step, batch in enumerate(train_dataloader):
            # Unpack the inputs and labels
            inputs, attention_mask, labels = batch

            # move tensors to GPU take advantage of the parallel computing power of a GPU to speed up the training
            # process.
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

            if step % print_frequency == 1:
                print(f'Epoch: {epoch + 1}, Batch: {step}, Loss: {running_loss / print_frequency}')
                running_loss = 0
        # Validate the model after each epoch
        _, val_f1_score = evaluate_model(model, val_dataloader, device)
        print(f'Validation F1 Score: {val_f1_score:.4f}')
        # If current model is better than the best model, overwrite best model with it
        if(val_f1_score > best_score):
            checkpoint(model, 'models/best_model.pth')
            print('\nPrevious best model had an f1 score of ' + str(best_score) + '. Overwriting best model.')
            f = open('model/best_f1.txt', 'w')
            f.seek(0)
            f.truncate()
            f.write(str(val_f1_score))
            f.close()
            f = open('models/best_epoch.txt', 'w')
            f.seek(0)
            f.truncate()
            f.write(str(val_f1_score))
            f.close()
    resume(model, 'models/best_model.pth')


def evaluate_model(model, test_dataLoader, device):
    model.eval()

    with torch.no_grad():
        all_preds = []
        all_labels = []

        for batch in test_dataLoader:
            inputs, attention_mask, labels = batch

            # move tensors to GPU
            inputs, attention_mask, labels = inputs.to(device), attention_mask.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs, attention_mask=attention_mask)[0]
            logits = outputs.detach().cpu().numpy()
            preds = logits.argmax(axis=1)

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='binary')

    return all_preds, f1
