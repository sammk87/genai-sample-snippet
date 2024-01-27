import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def tokenize_text(text):
    return word_tokenize(text)

def clean_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Convert to lowercase
    text = text.lower()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word.lower() not in stop_words]

    return ' '.join(words)

def perform_stemming(text):
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in text.split()]
    return ' '.join(words)

def perform_lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in text.split()]
    return ' '.join(words)

# Example text
sample_text = "Natural Language Processing (NLP) is a fascinating field with various applications. It involves the interaction between computers and human language, enabling machines to understand, interpret, and generate human-like text."

# Tokenization
tokens = tokenize_text(sample_text)
print("Tokenization:", tokens)

# Text Cleansing
cleaned_text = clean_text(sample_text)
print("Cleaned Text:", cleaned_text)

# Stop Word Removal
text_without_stopwords = clean_text(sample_text)
print("Text without Stopwords:", text_without_stopwords)

# Stemming
stemmed_text = perform_stemming(cleaned_text)
print("Stemming:", stemmed_text)

# Lemmatization
lemmatized_text = perform_lemmatization(cleaned_text)
print("Lemmatization:", lemmatized_text)
