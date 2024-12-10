
    
    
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 12:36:53 2024

@author: sr322
"""

import streamlit as st
from PIL import Image
from IPython import get_ipython
from IPython.display import display
import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv("C:/Users/sr322/Downloads/amazon_product.csv")

# Preprocessing
df.drop('id', axis=1, inplace=True)


# Tokenizer and Stemmer setup
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english')
nltk.download('punkt')

def tokenize_stem(text):
    tokens = nltk.word_tokenize(text.lower())
    stemmed = [stemmer.stem(w) for w in tokens]
    return " ".join(stemmed)

df['stemmed_tokens'] = df.apply(lambda row: tokenize_stem(row['Title'] + " " + row['Description']), axis=1)


# TF-IDF Setup
tfidf = TfidfVectorizer(tokenizer=tokenize_stem)
tfidf_matrix = tfidf.fit_transform(df['stemmed_tokens'])

# Cosine Similarity Function
def calculate_cosine_sim(txt1, txt2):
    """Calculates cosine similarity between two texts."""
    query_vector = tfidf.transform([txt1])
    index = df[df['stemmed_tokens'] == txt2].index[0]
    product_vector = tfidf_matrix[index]
    return cosine_similarity(query_vector, product_vector)[0][0]

# New Cosine Similarity Function (Between Texts)
def cosine_sim(text1, text2):
    """Calculates cosine similarity between concatenated texts."""
    text1_concatenated = ' '.join(text1)
    text2_concatenated = ' '.join(text2)
    tfidf_matrix = tfidf.fit_transform([text1_concatenated, text2_concatenated])
    return cosine_similarity(tfidf_matrix)[0][1]

# Search Functionality
def search_product(query):
    """Searches for products based on a query."""
    stemmed_query = tokenize_stem(query)
    df['similarity'] = df['stemmed_tokens'].apply(lambda x: calculate_cosine_sim(stemmed_query, x))
    res = df.sort_values(by='similarity', ascending=False).head(10)[['Title', 'Description', 'Category']]
    return res

# Streamlit Web App
st.title("Search Engine and Product Recommendation System ON Am Data")
query = st.text_input("Enter Product Name")
submit = st.button('Search')

if submit:
    res = search_product(query)
    st.write(res)
    