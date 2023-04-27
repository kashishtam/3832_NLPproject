import torch
import torch.nn as nn

def train_model(model, num_epochs, dataLoader, criterion, optimizer, device):
    print_frequency = 10
    softmax = nn.Softmax(dim=1)

    for epoch in range(num_epochs):
        print('### Epoch: ' + str(epoch + 1) + ' ###')

        running_loss = 0.0

        model.train()

        for step, batch in enumerate(dataLoader):
            # Unpack the inputs and labels
            inputs, attention_mask, labels = batch

            # move tensors to GPU take advantage of the parallel computing power of a GPU to speed up the training process.
            inputs, attention_mask, labels = inputs.to(device), attention_mask.to(device), labels.to(device)

            # Clear the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs, attention_mask=attention_mask)[0]
            output_probs = softmax(outputs)

            # Compute the loss
            loss = criterion(output_probs.squeeze(1), labels)

            # Backward pass
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Update the running loss
            running_loss += loss.item()

            if step % print_frequency == 1:
                print(f'Epoch: {epoch + 1}, Batch: {step}, Loss: {running_loss/print_frequency}')
                running_loss = 0

