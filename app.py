import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle

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

# Streamlit app layout
st.title("Research Paper Keyword Extractor")

# File uploader for the CSV file
uploaded_file = st.file_uploader("Upload your CSV file with papers", type="csv")

if uploaded_file is not None:
    # Load the uploaded CSV
    df = pd.read_csv(uploaded_file)

    # Limit the dataset to 5000 rows (for performance reasons)
    df = df.iloc[:5000, :]

    # Show the first few papers' titles and abstracts
    st.subheader("Select a Paper to Analyze")
    selected_idx = st.selectbox("Choose a paper:", df.index, format_func=lambda x: df['title'][x])

    st.write(f"**Title**: {df['title'][selected_idx]}")
    st.write(f"**Abstract**: {df['abstract'][selected_idx]}")

    # Preprocess the text column (paper_text)
    st.write("Processing the paper text...")
    docs = df['paper_text'].apply(preprocess_text)

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
    keywords = get_keywords(selected_idx, docs, cv, tfidf_transformer, feature_names)
    for word, score in keywords.items():
        st.write(f"**{word}**: {score}")
