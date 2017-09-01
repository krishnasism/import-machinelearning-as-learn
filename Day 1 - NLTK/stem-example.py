from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

example_words =["meow","meowing","meowingtom","meows"]

for w in example_words:
    print(ps.stem(w))
    
new_text = "The call of a cat is meow. When the cat was hungry it was meowing. The name of the cat was meowingtom. This cat meows at night every day!"
words = word_tokenize(new_text)
for w in words:
    print(ps.stem(w))
                     