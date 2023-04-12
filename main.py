# only for testing purposes 

from DataLoader import CSVReader
from PreProcess import PreProcess

from transformers import BertTokenizer

reader = CSVReader("train/train.EN.csv")
reader.read_file()
text = reader.get_sentence_column(4)
print(text)

process = PreProcess()
processed_text = process.preprocess_text(text)
processed_text = process.clean_text(processed_text)
print(processed_text)

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the text
tokens = tokenizer.tokenize(processed_text)

# Convert the tokenized text into numerical inputs
input_ids = tokenizer.convert_tokens_to_ids(tokens)

# Print the tokenized text and numerical inputs
print(f"Tokenized text: {tokens}")
print(f"Numerical inputs: {input_ids}")



