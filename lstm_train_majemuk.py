import pandas as pd
import matplotlib.pyplot as plt
from keras.metrics import Precision, Recall
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np


# Load the dataset
file_path = 'dataset_majemuk_lstm.csv'
data = pd.read_csv(file_path)

# Tokenizing the characters in the words
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(data['katamajemuk'])
sequences = tokenizer.texts_to_sequences(data['katamajemuk'])
print(tokenizer.word_index)
# Padding sequences
max_length = max([len(seq) for seq in sequences])
X = pad_sequences(sequences, maxlen=max_length, padding='post')

# Encoding labels (1 for correct, 0 for incorrect)
y = np.where(data['label'] == 'correct', 1, 0)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Building the LSTM model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_length))
model.add(LSTM(200)) # Adjust the number of units based on complexity
model.add(Dense(1, activation='sigmoid')) # Output layer for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',Precision(), Recall()])

early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Train the model
model.fit(X_train, y_train, batch_size=64, epochs=10, callbacks=[early_stopping], validation_split=0.2) # Adjust epochs as needed

model.save('lstm_majemuk_model2.h5')
# Evaluate the model on the test set
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Test Recall: {test_recall}")
print(f"Test Precision: {test_precision}")
print(f"Test F1_score: {2*(test_precision*test_recall)/(test_precision+test_recall)}")

# Predict labels for the test set
y_pred_probs = model.predict(X_test)

# Convert probabilities to binary predictions
y_pred = (y_pred_probs > 0.5).astype(int)

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Display the confusion matrix using seaborn heatmap
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.show()

# Print classification report
print(classification_report(y_test, y_pred))