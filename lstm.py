import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


import numpy as np

file_path = 'all_word_dataset.csv'  # Replace with your file path
data = pd.read_csv(file_path)

model = load_model('lstm_model.h5')

# Parameters for preprocessing
max_length = max([len(word) for word in data['word']]) # Maximum length of a word


# Tokenizing the characters in the words
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(data['word'])
sequences = tokenizer.texts_to_sequences(data['word'])

import string
# Assuming 'tokenizer' is your character-level tokenizer used during training
def preprocess_sentence(sentence, tokenizer, max_length):
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()
    sentence = stopword.remove(sentence)
    sentence = sentence.lower()
    sentence = sentence.translate(str.maketrans("", "", string.punctuation.replace('-', '')))
    words = sentence.split()
    tokenized_words = tokenizer.texts_to_sequences(words)
    padded_words = pad_sequences(tokenized_words, maxlen=max_length, padding='post')
    return words, padded_words

sentence = 'sukarelaa tunanetraa tunawismaa tunanetr pancaindrs tanggungjawab batang lher mataair matakaki matakayu pancaindra sukarela rumahsakit kepalanegara lalala rumah skit makn makan rumah skit rmah sakit kaim thun baru akhir pkan akhr tahun olhraga'
words, preprocessed_sentence = preprocess_sentence(sentence, tokenizer, max_length)

predictions = model.predict(preprocessed_sentence)

# Determine a threshold for correctness, e.g., 0.5
threshold = 0.4
correctness = predictions > threshold
typowords = []
i = 0
for word, is_correct in zip(words, correctness):
    print(f"Word: {word}, Correct: {is_correct[0]}, {predictions[i]}")
    if is_correct[0] :
        pass
    else : typowords.append(word)
    i = i+1
print(typowords)

