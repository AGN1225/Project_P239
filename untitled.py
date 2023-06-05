import streamlit as st
import pickle
import numpy as np

# Load the saved model
with open('clsf.pkl', 'rb') as file:
    model = pickle.load(file)

# Create a function to predict the sentiment class
def predict_sentiment(tweet):
    # Preprocess the tweet (apply the same preprocessing steps used during training)
    # ...

    # Perform inference/prediction
    tweet = np.array(tweet).reshape(1, -1)
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