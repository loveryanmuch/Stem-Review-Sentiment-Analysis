import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from os.path import join, dirname, abspath

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Function Definitions
def load_data(file_path):
    try:
        return pd.read_csv("C:\\Users\\17789\\LHL\\Stem-Review-Sentiment-Analysis\\Data\\dataset.csv")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def clean_text(text, stop_words, lemmatizer):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.strip()
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]  # Remove stopwords and lemmatize
    return ' '.join(tokens)

def plot_review_length_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['review_length'], bins=50, kde=True)
    plt.title('Distribution of Review Lengths')
    plt.xlabel('Review Length')
    plt.ylabel('Frequency')
    plt.show()

def remove_length_outliers(df):
    Q1 = df['review_length'].quantile(0.25)
    Q3 = df['review_length'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df['review_length'] >= lower_bound) & (df['review_length'] <= upper_bound)]

# Main Script
if __name__ == "__main__":
    # Download necessary NLTK resources
    nltk.download('stopwords')
    nltk.download('wordnet')

    # File path
    file_path = join(dirname(abspath(__file__)), "Data", "dataset.csv")

    # Load dataset
    reviews_df = load_data(file_path)
    if reviews_df is None:
        exit()

    # Preprocessing
    reviews_df = reviews_df[['app_id', 'app_name', 'review_text', 'review_score']]
    reviews_df.dropna(subset=['review_text'], inplace=True)

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    reviews_df['cleaned_review_text'] = reviews_df['review_text'].apply(lambda x: clean_text(x, stop_words, lemmatizer))
    
    reviews_df['review_length'] = reviews_df['cleaned_review_text'].apply(lambda x: len(x.split()))

    # Plotting and Outlier Removal
    plot_review_length_distribution(reviews_df)
    reviews_df_filtered = remove_length_outliers(reviews_df)

    # Save the filtered data
    reviews_df_filtered.to_csv("C:\\Users\\17789\\LHL\\Stem-Review-Sentiment-Analysis\\Data\\cleaned_dataset.csv", index=False)