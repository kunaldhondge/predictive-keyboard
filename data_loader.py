import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')

# load data
with open('sherlock.txt', 'r', encoding='utf-8') as f:
    text = f.read().lower()

tokens = word_tokenize(text)
print("Total Tokens:", len(tokens))