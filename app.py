from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from keras.datasets import imdb
import numpy as np
from keras.preprocessing import sequence
import re

app = Flask(__name__)

model = load_model('SentimentAnalysisModel.keras')
word_index = imdb.get_word_index()

def preprocess_review(review):
    review = re.sub(r'[^\w\s]', '', review).lower()
    words = review.split()
    word_indices = [word_index.get(word, 2) for word in words]
    max_review_length = 500
    word_indices_padded = sequence.pad_sequences([word_indices], maxlen=max_review_length)
    return word_indices_padded



@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['data']
    input_text_padded = preprocess_review(input_text)
    prediction = model.predict(input_text_padded)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    probability = prediction[0][0]
    return render_template('result.html', sentiment=sentiment, probability=probability)

if __name__ == '__main__':
    app.run(debug=True)
