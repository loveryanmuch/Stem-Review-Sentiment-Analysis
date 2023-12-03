from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling1D

# Convert integer sequences to a numpy array
import numpy as np
input_ids = np.array(tokens['input_ids'])
attention_masks = np.array(tokens['attention_mask'])

X_train, X_test, y_train, y_test = train_test_split(
    input_ids, 
    df['sentiment_label'], 
    test_size=0.2, 
    random_state=42
)

# Load the BERT model
bert = TFBertModel.from_pretrained('bert-base-uncased')

# Build the model
input_ids = Input(shape=(50,), dtype='int32', name='input_ids')
attention_masks = Input(shape=(50,), dtype='int32', name='attention_masks')

bert_output = bert(input_ids, attention_mask=attention_masks)[1]
dropout = Dropout(0.3)(bert_output)
output = Dense(3, activation='softmax')(dropout) # Assuming 3 classes: positive, negative, neutral

model = Model(inputs=[input_ids, attention_masks], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(
    [X_train, attention_masks[:len(X_train)]], 
    y_train, 
    batch_size=32, 
    validation_split=0.1, 
    epochs=3
)

model.evaluate([X_test, attention_masks[len(X_train):]], y_test)