import pandas as pd
import numpy as np
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from transformers import TFBertForSequenceClassification
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch

# Load dataset
df = pd.read_csv("C:\\Users\\17789\\LHL\\Stem-Review-Sentiment-Analysis\\Data\\Modeling.csv")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_reviews(reviews, tokenizer, max_length=128):
    return tokenizer.batch_encode_plus(
        reviews, 
        max_length=max_length, 
        padding='max_length', 
        truncation=True,
        return_attention_mask=True
    )

encoded_reviews = encode_reviews(df['review_text'].tolist(), tokenizer)
input_ids = np.array(encoded_reviews['input_ids'])
attention_masks = np.array(encoded_reviews['attention_mask'])
labels = df['sentiment'].values

X_train, X_test, y_train, y_test = train_test_split(
    input_ids, 
    labels, 
    test_size=0.2, 
    random_state=42
)

class BertHyperModel(HyperModel):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def build(self, hp):
        # Load the pre-trained BERT model
        model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.num_classes)

        # Hyperparameters to tune
        learning_rate = hp.Choice('learning_rate', values=[1e-5, 2e-5, 3e-5])
        
        # Compile the model
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss=BinaryCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        return model

hypermodel = BertHyperModel(num_classes=2)

tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='hyperparam_tuning',
    project_name='bert_sentiment_analysis'
)

tuner.search(X_train, y_train, epochs=3, validation_data=(X_test, y_test))

# Retrieve the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate the best model
loss, accuracy = best_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")

# Save the best model
best_model.save("C:\\Users\\17789\\LHL\\Stem-Review-Sentiment-Analysis\\Data\\Best_model.csv")
