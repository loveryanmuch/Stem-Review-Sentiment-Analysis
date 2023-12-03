import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string

# Make sure to download these resources if you haven't already
nltk.download('stopwords')
nltk.download('wordnet')

# Load your dataset
reviews_df = pd.read_csv("C:\\Users\\17789\\LHL\\Stem-Review-Sentiment-Analysis\\Data\\dataset.csv")

# Drop unnecessary columns
reviews_df = reviews_df[['review_text', 'review_score']]  # Keep only the columns you need

# Handle missing values
reviews_df.dropna(subset=['review_text'], inplace=True)  # Drop rows where review_text is missing

# Text preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.strip()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize
    return ' '.join(tokens)

# Apply the clean_text function to the review_text column
reviews_df['cleaned_review_text'] = reviews_df['review_text'].apply(clean_text)
