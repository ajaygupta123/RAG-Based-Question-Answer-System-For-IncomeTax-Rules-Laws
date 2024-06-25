import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TextCleaner:
    def __init__(self, remove_stopwords=True, use_stemming=True):
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else None
        self.ps = PorterStemmer() if use_stemming else None
        self.lemmatizer = WordNetLemmatizer() if not use_stemming else None

    def clean_text(self, text):
        # Remove non-text elements (e.g., numbers, special characters)
        text = re.sub(r'[^A-Za-z\s]', '', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Tokenize text
        words = word_tokenize(text)

        # Remove stop words
        if self.remove_stopwords:
            words = [word for word in words if word.lower() not in self.stop_words]

        # Remove punctuation
        words = [word for word in words if word not in string.punctuation]

        # Convert to lowercase
        words = [word.lower() for word in words]

        # Remove special characters and digits
        words = [word for word in words if word.isalpha()]

        # Apply stemming or lemmatization
        if self.use_stemming:
            words = [self.ps.stem(word) for word in words]
        else:
            words = [self.lemmatizer.lemmatize(word) for word in words]

        # Join the cleaned words back into a single string
        cleaned_text = ' '.join(words)

        return cleaned_text

# # Example usage
# if __name__ == "__main__":
#     text_cleaner = TextCleaner(remove_stopwords=True, use_stemming=True)
#     pdf_text = "Your long document text goes here. This text will be split into manageable chunks for processing. Each chunk will be of a size that fits within the model's maximum token limit. Overlapping tokens ensure context is preserved across chunks."

#     cleaned_text = text_cleaner.clean_text(pdf_text)
#     print("Cleaned Text:", cleaned_text[:500])  # Print the first 500 characters of the cleaned text
