import streamlit as st
import pickle
import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Load the saved model
with open('clsf.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the saved Word2Vec model
w2v_model = Word2Vec.load('word2vec.model')

# Create a function to preprocess the tweet
def preprocess_tweet(tweet):
    tokenized_tweet = tweet.split()
    embeddings = [w2v_model.wv[word] for word in tokenized_tweet if word in w2v_model.wv]
    if embeddings:
        tweet_embedding = np.mean(embeddings, axis=0)
    else:
        tweet_embedding = np.zeros(100)  # Use zero vector if no word embeddings found
    return tweet_embedding

# Create a function to predict the sentiment class
def predict_sentiment(tweet):
    # Preprocess the tweet
    tweet_embedding = preprocess_tweet(tweet)

    # Perform inference/prediction
    tweet = np.array(tweet_embedding).reshape(1, -1)
    sentiment = model.predict(tweet)

    return sentiment

# Set up the Streamlit app
def main():
    # Set a title for the app
    st.title('Sentiment Analysis')

    # Add a text input field for entering the tweet
    tweet_input = st.text_input('Enter a tweet')

    # Add a button to trigger the prediction
    if st.button('Predict'):
        # Call the predict_sentiment function
        sentiment = predict_sentiment(tweet_input)

        # Display the predicted sentiment class
        st.write('Predicted Sentiment:', sentiment)

# Run the app
if __name__ == '__main__':
    main()
