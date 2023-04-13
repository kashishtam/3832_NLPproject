import csv

class CSVReader_train:
    def __init__(self, filename):
        self.filename = filename
        # create empty lists for each column in the CSV file
        self.sentences = [] #sarcastic sentences
        self.rephrases = [] #rephrased evaluation
        self.sarcastic = [] # 1 is sarcastic and 0 is non-sarcastic
    def read_file(self):
        # open the CSV file and read its contents
        with open(self.filename, 'r', encoding='utf8') as file:
            reader = csv.reader(file)
            for row in reader:
                # append the value of each column to its corresponding list
                self.sentences.append(row[1])
                self.sarcastic.append(row[2])
                self.rephrases.append(row[3])
    def get_sentence_column(self,idx):
        return self.sentences[idx]
    def get_rephrase_column(self,idx):
        return self.rephrases[idx]
    def get_sarcastic_column(self,idx):
        return self.sarcastic[idx]


class CSVReader_test:
    def __init__(self, filename):
        self.filename = filename
        # create empty lists for each column in the CSV file
        self.sentences = [] #sarcastic sentences
        self.sarcastic = [] # 1 is sarcastic and 0 is non-sarcastic
    def read_file(self):
        # open the CSV file and read its contents
        with open(self.filename, 'r', encoding='utf8') as file:
            reader = csv.reader(file)
            for row in reader:
                # append the value of each column to its corresponding list
                self.sentences.append(row[0])
                self.sarcastic.append(row[1])
    def get_sentence_column(self,idx):
        return self.sentences[idx]
    def get_sarcastic_column(self,idx):
        return self.sarcastic[idx]