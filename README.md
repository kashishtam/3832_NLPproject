# 3832_NLPproject

Team members: Brendan Lancaster, Kashish Tamrakar, Tyler Rayborn

## Project Description

Our project is a Sarcasm Detection model using a pre-trained BERT model. The goal of the project is to train a model that can accurately identify whether a given sentence is sarcastic or not. We are using a pre-trained BERT model as our base model and fine-tuning it on the iSarcasmEval dataset. We are splitting the dataset into training and validation sets and using the training set to train the model and the validation set to evaluate its performance. We are using the F1 score as our evaluation metric. Finally, we are using checkpoints to save the model at regular intervals during training to enable us to resume training if necessary and to select the best-performing model.

## Project Files 

DataLoader.py: Loads data for training and testing

PreProcess.py: Cleans text in the data to remove emojis and other symbols that would interfere with reading data before using in training or evaluation

main.py : Trains or loads models from start to finish, calculates f1 scores and output for test dataset.

training.py: Contains code for training the model on the dataset and code for evaluating the f1 score of finished models

output.csv: Our generated output file with the texts and our model's evaluation on whether the text is sarcastic (1) or non-sarcastic (0)

train: Contains our training data

test: Contains our testing data

models: Contains best checkpointed model, best epoch, and best validation f1 score

## How to use

python main.py

You will be prompted to either use the best checkpointed model(file must exist in models/best_model.pth) or train a new model. Enter 'Y' to use the best checkpointed model or 'N' to train a new model.


