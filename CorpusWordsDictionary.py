import re
import json

# A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.  
LETTERS_NODIA = "acdeeinorstuuyz"
LETTERS_DIA = "áčďéěíňóřšťúůýž"
DIA_TO_NODIA = str.maketrans(LETTERS_DIA , LETTERS_NODIA)

def create_dictionary(file_name: str):
    dictionary = dict()
    with open(file_name , "r", encoding="utf-8") as f:
        for line in f:
            for word in re.findall(r"\w+",line):
                without_diacritics = word.translate(DIA_TO_NODIA)
                if without_diacritics in dictionary.keys():
                    dictionary[without_diacritics].add(word)
                else:
                    dictionary[without_diacritics] = {word}

    print("Dictionary created...")
    print("Found {} undiacritized unique words...".format(len(dictionary)))
    return dictionary
# Open a file for writing
# with open('dictionary.json', 'w') as F:
#     # Write the dictionary to the file in JSON format
#     json.dump(dictionary , F)

# with open('dictionary.json', 'r') as F:
#     # Read the dictionary from the file
#     dictionary = json.load(F)


