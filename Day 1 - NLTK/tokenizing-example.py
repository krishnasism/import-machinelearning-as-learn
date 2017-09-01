"""
Created on Fri Sep  1 08:46:40 2017

@author: Krishnasis
"""

from nltk.tokenize import sent_tokenize, word_tokenize


#tokenizing - word 
#           - sentence
#lexicon & corporas
#corpora - body of text ex:medical journals, presidential speeches 
#lexicon - words and their meanings

example_text =  "Hello, I am K. Mandal. I am learning Machine Learning in Python. Python is awesome! The sky looking dull today."
print("Sentences\n")
print(sent_tokenize(example_text)) #tokenize into sentences
print("Words\n")
print(word_tokenize(example_text)) #tokenize into words
print()
