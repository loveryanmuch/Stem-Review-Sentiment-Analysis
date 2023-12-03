from transformers import pipeline, BertTokenizer, TFBertModel, TFBertForSequenceClassification
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import tensorflow as tf
import pandas as pd

# Load dataset
df = pd.read_csv("C:\\Users\\17789\\LHL\\Stem-Review-Sentiment-Analysis\\Data\\dataset.csv")

# Load a pre-trained sentiment analysis pipeline
sentiment_pipeline = pipeline('sentiment-analysis')

def label_reviews(reviews):
    labeled_reviews = []
    for review in reviews:
        result = sentiment_pipeline(review)[0]
        label = 1 if result['label'] == 'POSITIVE' else 0  # Assuming binary classification for simplicity
        labeled_reviews.append((review, label))
    return labeled_reviews

# Assuming 'df' is your DataFrame and 'review' is the column with text data
labeled_data = label_reviews(df['review'].tolist())
labeled_df = pd.DataFrame(labeled_data, columns=['review', 'sentiment'])

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

def encode_reviews(reviews, tokenizer, max_length=128):
    encoded = tokenizer.batch_encode_plus(
        reviews, 
        max_length=max_length, 
        padding='max_length', 
        truncation=True,
        return_attention_mask=True
    )
    return encoded

def extract_bert_embeddings(encoded, model):
    input_ids = np.array(encoded['input_ids'])
    attention_mask = np.array(encoded['attention_mask'])

    # Get the embeddings from the last hidden state
    with tf.GradientTape() as tape:
        embeddings = model(input_ids, attention_mask=attention_mask)[0][:,0,:].numpy()
    return embeddings

# Encode and extract embeddings
encoded_reviews = encode_reviews(labeled_df['review'].tolist(), tokenizer)
embeddings = extract_bert_embeddings(encoded_reviews, bert_model)

# Split data
X_train, X_test, y_train, y_test = train_test_split(embeddings, labeled_df['sentiment'], test_size=0.2, random_state=42)

# Initialize and train a logistic regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Evaluate the model
accuracy = lr_model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy}")

# Load the BERT model for sequence classification
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # Assuming 3 classes: positive, negative, neutral

# Compile the model
optimizer = Adam(learning_rate=3e-5)
model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])

# Prepare training data (assuming you have X_train, y_train, X_val, y_val prepared)
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(32)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)

# Train the model
model.fit(train_dataset, epochs=3, validation_data=val_dataset)

# Predict on test set
y_pred = model.predict(X_test).logits
y_pred = np.argmax(y_pred, axis=1)

# Compute metrics
print(classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive']))

# Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)