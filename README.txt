NAME: Diacritization
AUTHOR: Jakub Hajko
YEAR: 2023


INTRODUCTION:
This project is designed to perform diacritization for Czech text. It takes Czech text without diacritics as input and outputs the same text with diacritics. The corpus for this project was created from data obtained from all articles from the website vesmir.cz. The code is written in Python and is divided into three files. The file DataGetter.py contains the script to read the articles from vesmir.cz and create the corpus. The file CorpusWordsDictionary.py creates a dictionary used during the prediction phase. The file Diacritization.py contains the classification algorithm used to perform the diacritization.


HOW TO USE:
a) Run script RUN.sh (args pre-set in the script). It will: 1.) Download the data corpus. 2.) Train the model and create the dictionary (save the model to the model_3 file). 3.) evaluate on evaluation set
b) Run DataGetter.py (if u don't have file "VesmirCorpus.txt" already, if u have the file call of the DataGetter.py will only rewrite the file), Run the file Diacritization.py (set args first, default settings visible in the file).


IMPORTANT "args" FOR THE Diacritization.py:

--lines_number: This argument specifies the number of lines from the corpus that will be used to train the model. The default value is 1500. Note that increasing this number may improve the performance of the model, but it will also increase the training time.

--corpus_file_name: This argument specifies the name of the corpus text file. The default value is "VesmirCorpus.txt".

--model_file_name: This argument specifies the name of the file where the trained model will be stored. The default value is "model_2".

--train_model: This argument specifies whether to train a new model or use an existing one. The default value is True, which means that a new model will be trained.


MODELS:
model_1 - mlp that was trained for ~31000 seconds on the full corpus, dictionary was created from the full corpus
model_2 - lr that was trained for ~40 seconds on first 1500 lines of the corpus, dictionarywas created from the full corpus (8461 lines)
model_3 - Can be trained and evaluated with script RUN.sh