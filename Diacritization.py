#!/usr/bin/env python3
import argparse
import lzma
import pickle
import re

import numpy as np
from typing import Tuple
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import CorpusWordsDictionary

parser = argparse.ArgumentParser()
parser.add_argument("--hidden_layers", nargs="+", default=[100], type=int, help="Hidden layer sizes")
parser.add_argument("--model", default="lr", type=str, help="Model to use lr or mlp")
parser.add_argument("--solver", default="saga", type=str, help="LR solver")
parser.add_argument("--max_iter", default=100, type=int, help="Max iters")
parser.add_argument("--train_model", default=False, type=bool, help="Train new model")
parser.add_argument("--model_file_name", default="model_2", type=str, help="Name of the file where the model is stored")
parser.add_argument("--corpus_file_name", default="VesmirCorpus.txt", type=str, help="Name of corpus text file")
parser.add_argument("--eval_file_name", default="diacritics-etest.txt", type=str, help="Name of evaluation corpus text file")
parser.add_argument("--lines_number", default=1500, type=int, help="Numer of lines from the corpus that will be used to train the model")
parser.add_argument("--INPUT", default="file", type=str, help="Input for the evaluation of the model is 'file' or 'stdin'")

# A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
diacritizable = {'i','y','c','n','a','s','d','l','e','r','t','u','o','z'}
LETTERS_NODIA = "acdeeinorstuuyz"
LETTERS_DIA = "áčďéěíňóřšťúůýž"
DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper() , LETTERS_NODIA + LETTERS_NODIA.upper())


def extract_features(file_name: str, max_number_of_lines: int) -> Tuple[np.ndarray, np.ndarray]:
    data = [] # Extracted data
    labels = [] # Extracted targets

    line_counter = 0
    with open(file_name,"r",encoding="utf-8") as file:
        for line in file:

            #count lines and if we reach max then stop reading 
            line_counter +=1
            if line_counter > max_number_of_lines:
                break

            # Strip the line from diacritics 
            striped_line = line.translate(DIA_TO_NODIA).rstrip("\n")
            for i in range(len(striped_line)):
                # Go through all characters and if it is diacritizable, extract features and add it to "data"
                if striped_line[i] in diacritizable:
                    n = len(striped_line)-1
                    # Take 5 characters before and 5 characters after the character on index "i" (if less than 5 characters before or after, fill with spaces)
                    feature_vector = " " * -min(0,i-5) + striped_line[max(0,i-5):min(n+1,i+6)] + " " * -min(0, n - (i+5)) 
                    feature_vector =  [a for a in feature_vector]

                    # Add n-grams of size 2
                    if i == 0:
                        feature_vector.append(" " + striped_line[i])
                    else:
                        feature_vector.append(striped_line[i-1:i+1])

                    if i == n:
                        feature_vector.append(striped_line[i] + " ")
                    else:
                        feature_vector.append(striped_line[i:i+2]) 

                    # Add the feature vector to "data" and its target to "labels" 
                    data.append(feature_vector)
                    labels.append(letter_to_target(line[i]))
    return (np.array(data),np.array(labels))

def extract_features_evaluation(text: str) -> np.ndarray:
    data = [] # Extracted data

    for line in text.split("\n"):
        for i in range(len(line)):
            # Go through all characters and if it is diacritizable, extract features and add it to "data"
                if line[i] in diacritizable:
                    n = len(line)-1
                    # Take 5 characters before and 5 characters after the character on index "i" (if less than 5 characters before or after, fill with spaces)
                    feature_vector = " " * -min(0,i-5) + line[max(0,i-5):min(n+1,i+6)] + " " * -min(0, n - (i+5)) 
                    feature_vector =  [a for a in feature_vector]

                    # Add n-grams of size 2
                    if i == 0:
                        feature_vector.append(" " + line[i])
                    else:
                        feature_vector.append(line[i-1:i+1])

                    if i == n:
                        feature_vector.append(line[i] + " ")
                    else:
                        feature_vector.append(line[i:i+2]) 

                    # Add the feature vector to "data"  
                    data.append(feature_vector)
                    
    return np.array(data)


def create_model():
    return Pipeline([
        ("one-hot", OneHotEncoder(handle_unknown="ignore")),
        ("estimator", {
            "lr": LogisticRegression(solver=args.solver, multi_class="multinomial", max_iter=args.max_iter, verbose=1),
            "mlp": MLPClassifier(hidden_layer_sizes=args.hidden_layers, max_iter=args.max_iter, verbose=1),
        }[args.model]),
    ])

def save_model(file_name, model):
    with lzma.open(file_name, "wb") as model_file: 
        pickle.dump(model , model_file)

def load_model(file_name):
    with lzma.open(file_name , "rb") as model_file:
        model = pickle.load(model_file)
    return model 

def letter_to_target(letter):
    if letter in "áéíóúý":
        return 1
    if letter in "čďěňřšťůž":
        return 2
    return 0

def compose(letter, diacritics_type):
    if diacritics_type == 0:
        return letter
    if diacritics_type == 1:
        index = "aeiouy".find(letter)
        return "áéíóúý"[index] if index >= 0 else letter
    if diacritics_type == 2:
        index = "cdenrstuz".find(letter)
        return "čďěňřšťůž"[index] if index >= 0 else letter
    
def word_likelihood(word , predictions, diacritizable_indices):
    likelihood = 0
    k = 0
    for i in diacritizable_indices:
        j = letter_to_target(word[i])
        if predictions[k,j] == 0:
            return float("-inf")
        likelihood += np.log(predictions[k,j])
        k+=1
    return likelihood


def main(args):
    data , labels = extract_features(file_name=args.corpus_file_name,max_number_of_lines=args.lines_number)
    dictionary = CorpusWordsDictionary.create_dictionary(args.corpus_file_name)

    if args.train_model: # We want to train new model
        model = create_model()
        model.fit(data,labels)
        save_model(args.model_file_name, model)
        print("Model trained and saved...")
    else: # We want to load already trained model
        model = load_model(args.model_file_name)
        print("Model loaded...")

    # Decide whether input from 'file' or 'stdin'
    if args.INPUT == 'file': 
        # Read evaluation file 
        with open(args.eval_file_name,"r",encoding="utf-8") as f:
            target_text = f.read()
    elif args.INPUT == 'stdin':
        print("--------Add text you want to diacritize(at the end press ctrl+D to signal end of the text)--------")
        target_text = ""
        while True:
            try:
                line = input()
            except EOFError:
                break
            target_text += line + "\n"

    # Test text in lower case without diacritics ready to create test_data
    test_text = target_text.lower().translate(DIA_TO_NODIA)
    diacritizable_indices = np.array([i for i in range(len(test_text)) if test_text[i] in diacritizable]) #indices of diacritizable characters in "test_text"
    test_data = extract_features_evaluation(test_text)
    
    
    predictions = model.predict_log_proba(test_data) # Predict

    index = 0
    new_text = list(test_text)
    for word in re.findall(r"\w+",test_text):
        diacritizable_count = sum([1 for c in word if c in diacritizable]) # Count diacritizable characters in word
        word_diacritizable_indices = [i for i in range(len(word)) if word[i] in diacritizable] # Indices of diacritizable characters in "word"
        indices_list = [i for i in range(index,(index+diacritizable_count))] # List that can be used to index in "predictions" and "diacritization_indices"
        index += diacritizable_count

        if word in dictionary.keys():
            if len(dictionary[word]) == 1:
                # Only one word in dictinary, copy it as the word prediction
                new_word = next(iter(dictionary[word])) # Get the single word
                
            elif len(dictionary[word]) > 1:
                # Compute likelyhood
                possible_words = list(dictionary[word]) # What word options we have in dictionary
                possible_words_likelihoods = [word_likelihood(w,predictions[indices_list],word_diacritizable_indices) for w in possible_words] # For each possible word compute likelihood
                new_word = possible_words[possible_words_likelihoods.index(max(possible_words_likelihoods))] # Choose the word with the best likelihood
            else:
                print("Should not happen")
                break

            # Copy all diacritizable characters from "new_word" to coresponding place in "new_text
            for j in range(diacritizable_count):
                    new_text[diacritizable_indices[indices_list[j]]] = new_word[word_diacritizable_indices[j]] 

        else:
            # Word not in dictionary take model prediction
            for i in indices_list:
                new_text[diacritizable_indices[i]] = compose(letter=new_text[diacritizable_indices[i]] ,diacritics_type=list(predictions[i]).index(max(predictions[i])))

    # new_text is in a lower case, map it back to normal acording to target
    for i in range(len(new_text)):
        if target_text[i].isupper():
            new_text[i] = new_text[i].upper()
            
    correct = 0
    total = 0
    for i in range(len(target_text)):
        if new_text[i] == target_text[i]:
            if target_text[i] == " " or target_text[i] == "\n":
                pass
            else: 
                correct += 1
                total += 1
        else:
            total += 1
    print("Non-white character accuracy: " + str(correct/total * 100) + " %")
    print("---------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    result = "".join(new_text)
    print(result)
    print("---------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)