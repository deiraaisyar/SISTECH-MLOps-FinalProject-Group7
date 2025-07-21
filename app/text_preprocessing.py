import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pandas as pd
import nltk

nltk.download('wordnet')
nltk.download('stopwords')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')

def lowering(text: str) -> str:
    text = text.lower()
    return text

def remove_punctuation_and_symbol(text: str) -> str:
    text = re.sub(r'[^\w\s]', '', text)
    return text

def stopword_removal(text: str) -> str:
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# def stemming(text: str) -> str:
#     text = " ".join([stemmer.stem(word) for word in text.split()])
#     return text

def lemmatization(text: str) -> str:
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_html_tags(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)
    return text

def preprocessing(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = remove_html_tags(text)
    text = lowering(text)
    text = remove_punctuation_and_symbol(text)
    text = stopword_removal(text)
    text = lemmatization(text)
    return text