# 3832_NLPproject

Team members: Brendan Lancaster, Kashish Tamrakar, Tyler Rayborn

## Project Description

DataLoader.py: loads data for training and testing

PreProcess.py: preprocesses text and clean text to remove emojis

util.py : calculates relevant metrics for the dataset

model.py : contains our model and forward function

main.py : runs everything from start to finish. However, parts are missing and are yet to be completed. They are commeneted with TODOs

test.py : testing for loading data, preprocessing data, tokenizing, and converting to numerical inputs

## How to use (thus far)

python test.py

## What it does

Based on the input text, it returns a preprocessed text and then its tokenized text and then its numerical inputs

## What we needs to be done 

Complete main.py by using what we have in test.py for the whole dataset and pass it through our model SarcasmDetector. Then, use a training loop to iterate through data in batches. 

After training, we will compute a performance metric with F1 score defined in util.py. We will thne fine-tune the model by adjusting its hyperparameters
