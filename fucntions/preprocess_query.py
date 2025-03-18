import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import contractions
from nltk.corpus import wordnet

# Define a function to convert mixed case to lower case
def convert_mixed_case_to_lower(x):
    if isinstance(x, str):
        lines = x.split('\n')
        new_lines = []
        for line in lines:
            words = line.split()
            new_words = []
            for word in words:
                if word.isupper():
                    new_words.append(word.lower())
                else:
                    new_words.append(word.lower())
            new_lines.append(' '.join(new_words))
        return ' '.join(new_lines)
    return x

# # Function to remove special character or ID patterns
# def remove_special_ids(text, pattern=r'\b\w*\d\w*\b'):
#     if isinstance(text, str):
#         return re.sub(pattern, ' ', text)
#     return text

def remove_punctuation(text):
        text = re.sub(r'/', 'or', text)
        text = re.sub(r'&', 'and', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        # text = re.sub(r'(?<!\d)\.(?!\d)|[^\w\s.]', ' ', text)
        return text

def tokenize(text):
    return word_tokenize(text)

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

def stemming(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]

def lemmatization(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]

def expand_contractions(text):
    return contractions.fix(text)

def remove_numbers(text):
    return re.sub(r'\d+', ' ', text)

def remove_extra_whitespace(text):
    return ' '.join(text.split())


def preprocess(text):
    text = convert_mixed_case_to_lower(text)
    # text = remove_special_ids(text)
    text = expand_contractions(text)
    text = remove_punctuation(text)
    text = remove_extra_whitespace(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatization(tokens)  # or use stemming(tokens)
    return ' '.join(tokens)

# Define the query and expand it
def expand_query(query):
    synonyms = set()
    for word in query.split():
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
    return query + " " + " ".join(synonyms)


def validate_input(user_input):
    if not user_input.strip():
        return False, "Input cannot be empty。"
    # can add more rules
    return True, ""
