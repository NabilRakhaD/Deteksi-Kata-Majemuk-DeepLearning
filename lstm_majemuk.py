import pandas as pd
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

katamajemuk = pd.read_csv('daftarkatamajemuk.csv')
katamajemuk = katamajemuk['kata majemuk'].values.tolist()

# Separate words with whitespace and without whitespace into two lists
with_whitespace = [word for word in katamajemuk if ' ' in word]
without_whitespace = [word for word in katamajemuk if ' ' not in word]

# Remove whitespace and combine words
combined_words = [''.join(word.split()) for word in with_whitespace]

# Function to split words based on specified rules
def split_words(word):
    if word in ['tragikomedi', 'pancaindra']:
        # Split after the fifth letter for specific cases
        return [word[:5], word[5:]]
    elif word in ['cespleng', 'tokcer']:
        # Split after the third letter for specific cases
        return [word[:3], word[3:]]
    else:
        # Split after the fourth letter for other cases
        return [word[:4], word[4:]]

# Apply the split_words function to each word in the list
result = [split_words(word) for word in without_whitespace]

# Flatten the list of lists and join each sublist into a string
flattened_result = [' '.join(item) for item in result]

misspelled_words_katamajemuk = combined_words + flattened_result
correct_words_katamajemuk = with_whitespace + without_whitespace
katamajemuk_2kata = with_whitespace + flattened_result
katamajemuk_1kata = without_whitespace + combined_words

# Load the dataset
file_path = 'dataset_majemuk_lstm.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Load the model
model = load_model('lstm_majemuk_model.h5')

# Parameters for preprocessing
max_length = max([len(word) for word in data['katamajemuk']])  # Maximum length of a word

# Tokenizing the characters in the words
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(data['katamajemuk'])

# Function to preprocess the compound word
# Assuming 'tokenizer' is your character-level tokenizer used during training
def preprocess_compound_word(compound_word, tokenizer, max_length):
    # Tokenize the word
    tokenized_word = tokenizer.texts_to_sequences([compound_word])
    # Pad the sequence
    padded_word = pad_sequences(tokenized_word, maxlen=max_length, padding='post')
    return padded_word

# Example compound word
# compound_word = "pancaindra"
# preprocessed_word = preprocess_compound_word(compound_word, tokenizer, max_length)


# Example compound word
dataset = pd.read_csv('daftarkatamajemuk.csv')
katamajemuk = dataset['kata majemuk'].values.tolist()
without_whitespace = [word for word in katamajemuk if ' ' not in word]
with_whitespace = [word for word in katamajemuk if ' ' in word]
combined_words = [''.join(word.split()) for word in with_whitespace]

compound_words = ['mata hari', 'bina raga', 'nara pidana', 'duka cita', 'suka cita', 'nara sumber', 'tuna busana', 'tuna karya', 'tuna netra', 'tuna rungu', 'tuna susila', 'tuna wicara', 'tuna wisma', 'suka ria', 'mala petaka', 'reka yasa', 'suka rela', 'tragi komedi', 'ces pleng', 'tok cer', 'olah raga', 'kaca mata', 'panca indra']
threshold = 0.4
x = katamajemuk_1kata[90:110]
for word in x:
    preprocessed_word = preprocess_compound_word(word, tokenizer, max_length)
    prediction = model.predict(preprocessed_word)
    is_correct = prediction[0, 0] > threshold
    print(f"Word: {word}, Correct: {is_correct}, Probability: {prediction[0, 0]}")

# from keras.models import load_model
# from keras.preprocessing.sequence import pad_sequences
# from keras.preprocessing.text import Tokenizer
# import pandas as pd
# import numpy as np

# # Load the dataset
# file_path = 'dataset_majemuk_lstm_new.csv'  # Replace with your file path
# data = pd.read_csv(file_path)

# # Parameters for preprocessing
# max_length = max([len(word) for word in data['katamajemuk']])

# # Tokenizing the characters in the words
# tokenizer = Tokenizer(char_level=True)
# tokenizer.fit_on_texts(data['katamajemuk'])
# sequences = tokenizer.texts_to_sequences(data['katamajemuk'])

# # Load the trained model
# model = load_model('lstm_majemuk_model.h5')

# def preprocess_input(compound_word, tokenizer, max_length):
#     # Tokenize and pad the compound word
#     sequence = tokenizer.texts_to_sequences([compound_word])
#     padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
#     return padded_sequence

# def prepare_features(root_word_label, first_root_word, second_root_word, tokenizer, max_length):
#     # Preprocess additional features
#     first_root_word_seq = tokenizer.texts_to_sequences([first_root_word])
#     first_root_word_padded = pad_sequences(first_root_word_seq, maxlen=max_length, padding='post')
    
#     second_root_word_seq = tokenizer.texts_to_sequences([second_root_word])
#     second_root_word_padded = pad_sequences(second_root_word_seq, maxlen=max_length, padding='post')
    
#     # Combine features
#     features = np.concatenate([[root_word_label], first_root_word_padded.flatten(), second_root_word_padded.flatten()])
#     return features.reshape(1, -1)

# # Example compound word to test
# compound_word = 'olahraga'  # Replace with your compound word
# root_word_label = 1 # 1 if root word, 0 otherwise
# first_root_word = 'olahraga'  # First part of compound word
# second_root_word = ''  # Second part of compound word

# # Preprocess the input word and features
# preprocessed_word = preprocess_input(compound_word, tokenizer, max_length)
# prepared_features = prepare_features(root_word_label, first_root_word, second_root_word, tokenizer, max_length)

# # Predict
# prediction = model.predict([preprocessed_word, prepared_features])
# print(f"Prediction for '{compound_word}':", prediction[0][0])

# # Interpret the result (using 0.5 as threshold)
# label = 'Correct' if prediction[0][0] > 0.4 else 'Incorrect'
# print(f"The compound word '{compound_word}' is predicted as: {label}")