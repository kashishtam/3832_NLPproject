# only for testing purposes 

from DataLoader import CSVReader_train
from DataLoader import CSVReader_test
from PreProcess import PreProcess

from transformers import BertTokenizer

# Plan to convert to dataframe and use Pytorch's dataloader
# Load training data
training_data = CSVReader_train("train/train.EN.csv")
training_data.read_file()

# Load test data
test_data = CSVReader_test("test/task_A_En_test.csv")
test_data.read_file()

# Test print for sentence in test data
test_text = test_data.get_sentence_column(1)
print(test_text)
assert(test_data.get_sarcastic_column(1) == '0')

# Test print for sentence in training data
text = training_data.get_sentence_column(4)
print(text)
assert(training_data.get_sarcastic_column(4) == '1')

# Preprocessing the text
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



