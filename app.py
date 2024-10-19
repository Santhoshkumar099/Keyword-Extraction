import streamlit as st
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load the trained models
with open('cv.pkl', 'rb') as f:
    cv = pickle.load(f)

with open('tfidf.pkl', 'rb') as f:
    tfidf_transformer = pickle.load(f)

with open('featurenames.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Initialize stop words
stop_words = set(stopwords.words('english'))
additional_words = ["fig", "figure", "image", "sample", "using", "show", "result", "large",
                    "also", "one", "two", "three", "four", "five", "seven", "eight", "nine"]
stop_words.update(additional_words)

# Preprocessing function
def preprocess_text(txt):
    txt = re.sub(r'<.*?>|[^a-z]', ' ', txt.lower())  # Lowercase, remove HTML tags and non-alphabetic characters
    tokens = [PorterStemmer().stem(word) for word in nltk.word_tokenize(txt) if word not in stop_words and len(word) >= 3]
    return ' '.join(tokens)

# Function to get keywords for a given text
def get_keywords(text):
    processed_text = preprocess_text(text)
    doc_vector = tfidf_transformer.transform(cv.transform([processed_text]))
    sorted_words = sorted(zip(doc_vector.tocoo().col, doc_vector.data), key=lambda x: (x[1], x[0]), reverse=True)[:10]
    return {feature_names[idx]: round(score, 3) for idx, score in sorted_words}

# Streamlit app
st.title("Keyword Extraction from Text")
st.write("Paste your text below to extract the top 10 keywords:")

# Text input
user_input = st.text_area("Input Text", height=300)

if st.button("Extract Keywords"):
    if user_input:
        keywords = get_keywords(user_input)
        st.write("**Top 10 Keywords:**")
        for word, score in keywords.items():
            st.write(f"**{word}**: {score}")
    else:
        st.warning("Please enter some text to extract keywords.")

