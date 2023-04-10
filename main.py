# only for testing purposes 

from DataLoader import CSVReader
from PreProcess import PreProcess

reader = CSVReader("train/train.EN.csv")
reader.read_file()
text = reader.get_sentence_column(1)
print(text)

process = PreProcess()
processed_text = process.preprocess_text(text)
print(processed_text)




