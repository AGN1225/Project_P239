import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model
with open('clsf.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the saved tokenizer (if applicable)
with open('tokenized.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Create a function to preprocess and encode the tweet
def preprocess_tweet(tweet):
    # Preprocess the tweet (apply the same preprocessing steps used during training)
    # ...

    # Tokenize the tweet using the saved tokenizer (or use your own tokenizer)
    tokenized_tweet = tokenizer.texts_to_sequences([tweet])

    # Pad the tokenized sequence to a fixed length
    max_sequence_length = 100  # Adjust this based on your training data
    padded_tweet = pad_sequences(tokenized_tweet, maxlen=max_sequence_length)

    return padded_tweet

# Create a function to predict the sentiment class
def predict_sentiment(padded_tweet):
    # Perform inference/prediction
    sentiment = model.predict(padded_tweet)[0]

    return sentiment

# Set up the Streamlit app
def main():
    # Set a title for the app
    st.title('Sentiment Analysis')

    # Add a text input field for entering the tweet
    tweet_input = st.text_input('Enter a tweet')

    # Add a button to trigger the prediction
    if st.button('Predict'):
        # Preprocess and encode the tweet
        padded_tweet = preprocess_tweet(tweet_input)

        # Call the predict_sentiment function
        sentiment = predict_sentiment(padded_tweet)

        # Display the predicted sentiment class
        st.write('Predicted Sentiment:', sentiment)

# Run the app
if __name__ == '__main__':
    main()
