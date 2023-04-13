import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import re
# uncoment if not installed

#nltk.download('omw-1.4')
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')

class PreProcess:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def preprocess_text(self, text):
        # Tokenize the text into words
        words = word_tokenize(text)

        # Convert the words to lowercase
        words = [word.lower() for word in words]

        # Remove stop words
        words = [word for word in words if word not in self.stop_words]

        # Lemmatize the words
        words = [self.lemmatizer.lemmatize(word, pos='v') for word in words]

        # Join the words back into a string
        preprocessed_text = ' '.join(words)

        return preprocessed_text
    
    def clean_text(self,text):
        # Code snippet from https://github.com/AbdelkaderMH/iSarcasmEval/blob/main/preprocessing.py
        # Remove shruggie from the text
        # text = emoji_pattern.sub(r'', text)
        text = re.sub(r'(?:@[\w_]+)', "", text)
        text = re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', "", text)
        text = text.replace('_', ' ')
        text = text.replace('#', ' ')
        text = text.replace(u"\u30c4",' ')
        text = text.replace('(',' ')
        text = text.replace(')',' ')
        text = text.replace(u"\u005c",' ')
        text = text.replace(u"\u002f",' ')

        return text
