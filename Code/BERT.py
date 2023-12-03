from transformers import pipeline
import tensorflow as tf
import pandas as pd

# Load dataset
df = pd.read_csv("C:\\Users\\17789\\LHL\\Stem-Review-Sentiment-Analysis\\Data\\dataset.csv")

# Create a sentiment analysis pipeline using the default BERT model for sentiment analysis
classifier = pipeline('sentiment-analysis')

# Function to classify review text and return only the label ('POSITIVE' or 'NEGATIVE')
def classify_review(review):
    return classifier(review)[0]['label']

# create a new column 'sentiment' with the sentiment label
reviews_df['sentiment'] = reviews_df['review_text'].apply(classify_review)

# Filter the DataFrame to include only rows where the sentiment is 'POSITIVE' or 'NEGATIVE'
reviews_df = reviews_df[reviews_df['sentiment'].isin(['POSITIVE', 'NEGATIVE'])]

# Map the sentiment labels to a binary target variable
# 'POSITIVE': 1, 'NEGATIVE': 0
target_map = {'POSITIVE': 1, 'NEGATIVE': 0}
reviews_df['target'] = reviews_df['sentiment'].map(target_map)

reviews_df.head()

# # Label encoding
# label_encoder = LabelEncoder()
# df['sentiment_label'] = label_encoder.fit_transform(df['sentiment_label'])

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# # Tokenize and encode sequences in the dataset
# tokens = tokenizer.batch_encode_plus(
#     df['review_text'].tolist(),
#     max_length = 50,
#     padding='max_length',
#     truncation=True
# )

