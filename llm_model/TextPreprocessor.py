import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import contractions
import string

class TextPreprocessor:
    def __init__(self):
        # Ensure required NLTK resources are downloaded
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')
        # Define the specific replacements
        self.replacements = {
                "_": " ",
                "/": " or ",
                "&": " and ",
            }


    def convert_mixed_case_to_lower(self, text):
        try:
            if isinstance(text, str):
                words = text.split()
                new_words = []
                for word in words:
                    if word.isupper() and len(word) > 1:
                        new_words.append(word)  # Retain the upper case word
                    else:
                        new_words.append(word.lower())  # Convert to lower case
                return ' '.join(new_words)
        except Exception as e:
            print(f"Error in convert_mixed_case_to_lower: {e}")
            return text

    def remove_punctuation(self, text):
        try:
            # Perform the specific replacements using re.sub()
            for old, new in self.replacements.items():
                text = re.sub(re.escape(old), new, text)
            # Define a regular expression pattern to match all punctuation
            punctuation_pattern = '[' + re.escape(string.punctuation) + ']'
            # Remove all punctuation using re.sub()
            text = re.sub(punctuation_pattern, '', text)
            return text
        except Exception as e:
            print(f"Error in remove_punctuation: {e}")
            return text

    def tokenize(self, text):
        try:
            return word_tokenize(text)
        except Exception as e:
            print(f"Error in tokenize: {e}")
            return text

    def remove_stopwords(self, tokens):
        try:
            stop_words = set(stopwords.words('english'))
            return [word for word in tokens if word not in stop_words]
        except Exception as e:
            print(f"Error in remove_stopwords: {e}")
            return tokens

    def get_wordnet_pos(self, word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {'J': wordnet.ADJ,
                    'N': wordnet.NOUN,
                    'V': wordnet.VERB,
                    'R': wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    def lemmatization(self, tokens):
        try:
            lemmatizer = WordNetLemmatizer()
            lemmatized_tokens = []
            for word in tokens:
                pos = self.get_wordnet_pos(word)
                lemmatized_word = lemmatizer.lemmatize(word, pos)
                lemmatized_tokens.append(lemmatized_word)
            return lemmatized_tokens
        except Exception as e:
            print(f"Error in lemmatization: {e}")
            return tokens

    def stemming(self, tokens):
        try:
            stemmer = PorterStemmer()
            return [stemmer.stem(word) for word in tokens]
        except Exception as e:
            print(f"Error in stemming: {e}")
            return tokens

    def expand_contractions(self, text):
        try:
            return contractions.fix(text)
        except Exception as e:
            print(f"Error in expand_contractions: {e}")
            return text

    def remove_extra_whitespace(self, text):
        try:
            return ' '.join(text.split())
        except Exception as e:
            print(f"Error in remove_extra_whitespace: {e}")
            return text

    def preprocess(self, text):
        try:
            text = self.convert_mixed_case_to_lower(text)
            text = self.expand_contractions(text)
            text = self.remove_punctuation(text)
            text = self.remove_extra_whitespace(text)
            tokens = self.tokenize(text)
            tokens = self.remove_stopwords(tokens)
            tokens = self.lemmatization(tokens)  # or use self.stemming(tokens)
            return ' '.join(tokens)
        except Exception as e:
            print(f"Error in preprocess: {e}")
            return text
