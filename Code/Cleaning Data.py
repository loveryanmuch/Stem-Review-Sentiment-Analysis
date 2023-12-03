import pandas as pd
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("C:\\Users\\17789\\LHL\\Stem-Review-Sentiment-Analysis\\Data\\dataset.csv")

# Basic text cleaning
def analyze_sentiment(reviews):
    # Load pre-trained model pipeline for sentiment analysis
    sentiment_pipeline = pipeline("sentiment-analysis")

    # Analyzing sentiment for each review
    results = []
    for review in reviews:
        result = sentiment_pipeline(review)
        results.append((review, result[0]['label'], result[0]['score']))

    return results

# Label encoding
label_encoder = LabelEncoder()
df['sentiment_label'] = label_encoder.fit_transform(df['sentiment_label'])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize and encode sequences in the dataset
tokens = tokenizer.batch_encode_plus(
    df['review_text'].tolist(),
    max_length = 50,
    padding='max_length',
    truncation=True
)

