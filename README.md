# 3832_NLPproject

Team members: Brendan Lancaster, Kashish Tamrakar, Tyler Rayborn

## Project Description

DataLoader.py: Loads data for training and testing

PreProcess.py: Cleans text in the data to remove emojis and other symbols that would interfere with reading data before using in training or evaluation

model.py : Contains our model and forward function

main.py : Trains or loads models from start to finish, calculates f1 scores and output for test dataset, saves if best model

training.py: Contains code for training the model on the dataset and code for evaluating the f1 score of finished models

## How to use

python main.py

You will be prompted to either use the best checkpointed model(file must exist in models/best_model.pth) or train a new model. Enter 'Y' to use the best checkpointed model or 'N' to train a new model.

## What it does

The program takes the training (and validation) data, processes it and returns tokenized text and its numerical inputs that the model is then trained on. 
Based on the input text, it returns a preprocessed text and then its tokenized text and then its numerical inputs

## What we needs to be done 

Complete main.py by using what we have in test.py for the whole dataset and pass it through our model SarcasmDetector. Then, use a training loop to iterate through data in batches. 

After training, we will compute a performance metric with F1 score defined in util.py. We will thne fine-tune the model by adjusting its hyperparameters
