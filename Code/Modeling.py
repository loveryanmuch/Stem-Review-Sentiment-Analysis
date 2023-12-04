import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import pipeline, BertTokenizer, TFBertModel, TFBertForSequenceClassification
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("C:\\Users\\17789\\LHL\\Stem-Review-Sentiment-Analysis\\Data\\Filtered_data.csv")

# Create a sentiment analysis pipeline using the default BERT model for sentiment analysis
sentiment_pipeline = pipeline('sentiment-analysis')

# Function to classify review text and return binary label
def classify_review(review):
    result = sentiment_pipeline(review)[0]
    return 1 if result['label'] == 'POSITIVE' else 0

# Apply the classification function
df['sentiment'] = df['review_text'].apply(classify_review)

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to encode reviews
def encode_reviews(reviews, tokenizer, max_length=128):
    return tokenizer.batch_encode_plus(
        reviews, 
        max_length=max_length, 
        padding='max_length', 
        truncation=True,
        return_attention_mask=True
    )

# Encode reviews
encoded_reviews = encode_reviews(df['review_text'].tolist(), tokenizer)
input_ids = np.array(encoded_reviews['input_ids'])
attention_masks = np.array(encoded_reviews['attention_mask'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(input_ids, df['sentiment'], test_size=0.2, random_state=42)

# Train a Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Evaluate the Logistic Regression model
accuracy = lr_model.score(X_test, y_test)
print(f"Logistic Regression Model Accuracy: {accuracy}")

# Load BERT model
bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # Binary classification

# Compile the BERT model
optimizer = Adam(learning_rate=3e-5)
bert_model.compile(optimizer=optimizer, loss=bert_model.compute_loss, metrics=['accuracy'])

# Train the BERT model
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, attention_masks[:len(X_train)], y_train)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, attention_masks[len(X_train):], y_test)).batch(32)

bert_model.fit(train_dataset, epochs=3, validation_data=test_dataset)

# Predict and Evaluate the BERT model
y_pred = bert_model.predict(test_dataset).logits
y_pred = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred, target_names=['NEGATIVE', 'POSITIVE']))
print(confusion_matrix(y_test, y_pred))

Modeling.save("C:\\Users\\17789\\LHL\\Stem-Review-Sentiment-Analysis\\Data\\Modeling.csv")
