import pandas as pd
from collections import defaultdict
import re
from flask import Flask, render_template, request
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import difflib
from nltk.tokenize import MWETokenizer

app = Flask(__name__, template_folder='index')


@app.route('/', methods=['GET', 'POST'])

def index() :
    
    if request.method == 'POST' :
        sentence = request.form['text']
        
        #PREPROCESS
        #jru bicara pemerintah lala mkan lala angkatan drat indonesia rmah sakit rumah skit adalah panca indrs sukarelaa seseorang jru bicara dari kaca mats lala gelap guleta di rumah mkan orangg tua olah tbuh rumahsakit ibukota
        #sukarelaa tunanetraa tunawismaa tunanetr pancaindrs 12 tanggungjawab batang leher mataair matakaki matakayu pancaindra sukarela rumahsakit kepalanegara lalala rumah skit makn makan rumah skit rmah sakit kaim thun baru akhir pkan akhr tahun olhraga

        factory = StopWordRemoverFactory()
        stopword = factory.create_stop_word_remover()
        sentence = stopword.remove(sentence)
        sentence = re.sub(r'\d', '',sentence)
        sentence = sentence.lower()
        sentence = sentence.translate(str.maketrans("", "", string.punctuation.replace('-', '')))
        sentence = ' '.join(sentence.split())

        #Predict typo word from a sentence

        # Example corpus
        corpus = pd.read_csv('katamajemuk_new copy.csv')
        corpus = corpus['katamajemuk'].values.tolist()

        def build_bigrams(corpus):
            bigrams = defaultdict(lambda: defaultdict(lambda: 0))
            for sentence in corpus:
                sentence = re.sub(r'[^\w\s]', '', sentence).lower().split()
                for i in range(len(sentence)-1):
                    bigram = sentence[i]
                    next_word = sentence[i+1]
                    bigrams[bigram][next_word] += 1
            return bigrams

        bigram_model = build_bigrams(corpus)

        def correct_misspelled_word(sentence, misspelled_word, bigram_model, option_correct_word):
            words = sentence.split()
            correction_options = option_correct_word
            misspelled_index = words.index(misspelled_word)
            best_correction = misspelled_word  # Default to original word
            highest_prob = 0

            if misspelled_index == 0:  # Misspelled word at the beginning
                for option in correction_options:
                    prob_following = bigram_model[option].get(words[1], 0)
                    if prob_following > highest_prob:
                        highest_prob = prob_following
                        best_correction = option
            elif misspelled_index == len(words) - 1:  # Misspelled word at the end
                for option in correction_options:
                    prob_preceding = bigram_model[words[-2]].get(option, 0)
                    if prob_preceding > highest_prob:
                        highest_prob = prob_preceding
                        best_correction = option
            else:  # Misspelled word in the middle
                preceding_word = words[misspelled_index - 1]
                following_word = words[misspelled_index + 1]
                for option in correction_options:
                    prob_preceding = bigram_model[preceding_word].get(option, 0)
                    prob_following = bigram_model[option].get(following_word, 0)
                    avg_prob = (prob_preceding + prob_following) / 2
                    if avg_prob > highest_prob:
                        highest_prob = avg_prob
                        best_correction = option

            # Replace the misspelled word with the best correction
            corrected_words = words.copy()
            corrected_words[misspelled_index] = best_correction
            return ' '.join(corrected_words)

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

        #misspelled_words_katamajemuk = combined_words + flattened_result
        #correct_words_katamajemuk = with_whitespace + without_whitespace
        katamajemuk_2kata = with_whitespace + flattened_result
        katamajemuk_1kata = without_whitespace + combined_words

        file_path = 'all_word_dataset.csv'  # Replace with your file path
        dataset = pd.read_csv('dataset_tiap_kata.csv')
        kata = dataset['kata'].values.tolist()
        data = pd.read_csv(file_path)

        model = load_model('lstm_model3.h5')

        # Parameters for preprocessing
        max_length = max([len(word) for word in data['word']]) # Maximum length of a word
        # vocab_size = len(set(''.join(data['word']))) # Number of unique characters
        # embedding_dim = 50 # Size of the embedding vector

        # Tokenizing the characters in the words
        tokenizer = Tokenizer(char_level=True)
        tokenizer.fit_on_texts(data['word'])

        # Assuming 'tokenizer' is your character-level tokenizer used during training
        def preprocess_sentence(sentence, tokenizer, max_length):
            words = sentence.split()
            tokenized_words = tokenizer.texts_to_sequences(words)
            padded_words = pad_sequences(tokenized_words, maxlen=max_length, padding='post')
            return words, padded_words

        words, preprocessed_sentence = preprocess_sentence(sentence, tokenizer, max_length)

        predictions = model.predict(preprocessed_sentence)

        #Get the predict typo words
        threshold = 0.4
        correctness = predictions > threshold
        typowords = []
        i = 0
        for word, is_correct in zip(words, correctness):
            if is_correct[0] :
                pass
            else : typowords.append(word)
            i = i+1
        print(typowords)


        # Load the dataset
        file_path = 'dataset kata benar.csv'  # Replace with your dataset path
        data = pd.read_csv(file_path)

        typo = []
        for words in typowords :
            # Generate the predicted word or sentence
            input_word = words  # replace with your word

            correct_words_list = set(data['word'])

            # Define the function to post-process predicted words using difflib
            def correct_predictions_with_options(words, correct_words_list, num_options=15):
                # Use difflib to get the closest matches to the word from the dictionary
                close_matches = difflib.get_close_matches(words, correct_words_list, n=num_options, cutoff=0.65)
                # If there are close matches, return them as options
                if close_matches :
                    return close_matches   
                # If no close matches are found, return the original predicted word as the only option
                return [words]

            # Get corrected options
            corrected_options = correct_predictions_with_options(words, correct_words_list)
            #print(f"Huruf yg di cek : {input_word}")
            #print(f"Original Predicted: {predicted_word}")
            print(f"Corrected Options: {corrected_options}")
            typo.append([input_word, corrected_options])

        #separate the typo compound words from the typo word
        x = []
        possible = []


        for a in range(len(typo)) :
            if typo[a][0] in katamajemuk_1kata  :
                pass
            else :
                for word in typo[a][1] :
                    if word in without_whitespace :
                        x.append(typo[a])
                        break
                    elif word in kata :
                        possible.append(typo[a])
                        break

        # Iterate through the words and replacements
        for word, replacements in x:
            # Create a regex pattern with word boundaries for the current word
            pattern = rf'\b{re.escape(word)}\b'
            
            # Choose a replacement from the list (for simplicity, it takes the first one)
            replacement = replacements[0]

            # Replace words using regex
            sentence = re.sub(pattern, replacement, sentence)


        for a in range(len(possible)) :
            corrected_sentence = correct_misspelled_word(sentence, possible[a][0], bigram_model, possible[a][1])
            sentence = corrected_sentence

        #Get the all compound words that have been corrected and not
        def mwetokenizer(sentence, katamajemuk_1kata) :

            # Convert the list of strings into a list of tuples
            mwe_tuples = [tuple(item.split()) for item in katamajemuk_2kata]

            # Initialize the tokenizer with your list of tuples
            tokenizer = MWETokenizer(mwe_tuples)

            # Tokenize the text
            tokens = tokenizer.tokenize(sentence.split())

            # Initialize two lists: one for words without underscores and one for words with underscores
            words_with_underscore = []

            # Check each word and separate them into the appropriate list
            for word in tokens:
                if '_' in word:
                    words_with_underscore.append(word)

            # Remove underscores from each word
            words_without_underscore = [word.replace('_', ' ') for word in words_with_underscore]

            words = sentence.split()
            found_compound_words = []

            i = 0
            while i < len(words):
                for katamajemuk_word in katamajemuk_1kata:
                    remaining_words = ' '.join(words[i:])
                    if remaining_words.startswith(katamajemuk_word):
                        found_compound_words.append(katamajemuk_word)
                        i += len(katamajemuk_word.split()) - 1  # Move the index to the end of the compound word
                        break
                i += 1

            return words_without_underscore,found_compound_words

        compoundword, singleword = mwetokenizer(sentence, katamajemuk_1kata)
        all = compoundword + singleword

        # PREDICT THE TYPO COMPOUND WORDS
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

        compound_words = all
        threshold = 0.4
        result = {}
        for word in compound_words:
            preprocessed_word = preprocess_compound_word(word, tokenizer, max_length)
            prediction = model.predict(preprocessed_word)
            is_correct = prediction[0, 0] > threshold
            suggestion = difflib.get_close_matches(word, katamajemuk, n=1, cutoff=0.65)
            #print(f"Word: {word}, Correct: {is_correct}, Probability: {prediction[0, 0]}, Suggestion:{suggestion}")
            result[word] = {
                'is_correct' : bool(is_correct),
                'suggestion' : suggestion[0]
            }

        #return result
        return render_template('result.html',  artikel=sentence, result=result)
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)  