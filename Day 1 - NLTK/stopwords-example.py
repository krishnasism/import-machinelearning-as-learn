from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sentence = "Hello, I am K. Mandal. I am learning Machine Learning in Python. Python is awesome! The sky looking dull today."
#create stop_words array with English Stopwords
stop_words = set(stopwords.words("english"))
#add or remove stopwords into our array
stop_words.add("Python")
stop_words.add("mamamia")
stop_words.remove("mamamia")
#diplay
print(stop_words)

words=word_tokenize(example_sentence)
filtered_sentence=[]
for w in words:
    if w not in stop_words:
        filtered_sentence.append(w)
       
print(filtered_sentence)