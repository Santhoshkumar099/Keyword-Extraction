import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle
import fitz  # PyMuPDF

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

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

# Get keywords for a given document index
def get_keywords(idx, docs, cv, tfidf_transformer, feature_names):
    doc_vector = tfidf_transformer.transform(cv.transform([docs[idx]]))
    sorted_words = sorted(zip(doc_vector.tocoo().col, doc_vector.data), key=lambda x: (x[1], x[0]), reverse=True)[:10]
    return {feature_names[idx]: round(score, 3) for idx, score in sorted_words}

# Function to extract text from PDF files
def extract_text_from_pdf(file):
    doc = fitz.open(file)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Streamlit app layout
st.title("Research Paper Keyword Extractor")

# File uploader for PDF or DOCX files
uploaded_file = st.file_uploader("Upload your PDF or DOCX file with papers", type=["pdf", "docx"])

if uploaded_file is not None:
    # Handle PDF files
    if uploaded_file.type == "application/pdf":
        st.write("Extracting text from PDF...")
        text = extract_text_from_pdf(uploaded_file)
    else:
        # You can implement a DOCX text extraction here
        st.write("DOCX files support coming soon!")

    # Preprocess the text
    st.write("Processing the paper text...")
    docs = [preprocess_text(text)]

    # Create or load the vectorizer and transformer models
    if 'cv.pkl' in st.session_state:
        cv = st.session_state['cv.pkl']
        tfidf_transformer = st.session_state['tfidf.pkl']
        feature_names = st.session_state['featurenames.pkl']
    else:
        cv = CountVectorizer(max_df=0.95, max_features=1000, ngram_range=(1, 3))
        word_count_vectors = cv.fit_transform(docs)
        
        tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True).fit(word_count_vectors)
        feature_names = cv.get_feature_names_out()
        
        st.session_state['cv.pkl'] = cv
        st.session_state['tfidf.pkl'] = tfidf_transformer
        st.session_state['featurenames.pkl'] = feature_names

    # Extract and display keywords
    st.write("Top 10 Keywords:")
    keywords = get_keywords(0, docs, cv, tfidf_transformer, feature_names)
    for word, score in keywords.items():
        st.write(f"**{word}**: {score}")

    # Extract and display keywords
    st.write("Top 10 Keywords:")
    keywords = get_keywords(selected_idx, docs, cv, tfidf_transformer, feature_names)
    for word, score in keywords.items():
        st.write(f"**{word}**: {score}")
