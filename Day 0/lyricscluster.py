# Import the data set into a Panda data frame
import pandas as pd
from io import StringIO
import requests
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


act = requests.get('https://docs.google.com/spreadsheets/d/1k3swWcY0SOtDodxz4XNDAz78bKm8qXYdv0iwcrLkrZk/export?format=csv')
dataact = act.content.decode('utf-8') # To convert to string for Stringio
frame = pd.read_csv(StringIO(dataact))
# Loop over each lyrics and append its content to a list of string named 'corpus'
corpus = []
for i in range(0, frame["Lyrics"].size):
    corpus.append(frame["Lyrics"][i])
# Create tf–idf matrix
vectorizer = TfidfVectorizer(stop_words = 'english', min_df = 0.2)
# min_df = 0.2 means that the term must be in at least 20% of the documents
X = vectorizer.fit_transform(corpus)

k = 3 #no. of clusters

dist = 1 - cosine_similarity(X)
# Run the algorithm kmeans
model = KMeans(n_clusters = k)
model.fit(X)

no_words = 10 # Number of words to print per cluster
order_centroids = model.cluster_centers_.argsort()[:, ::-1] # Sort cluster centers by proximity to centroid
terms = vectorizer.get_feature_names()
labels = model.labels_ # Get labels assigned to each data

print("Top terms per cluster:\n")
for i in range(k):

    print("Cluster %d songs:" % i, end='')
    for title in frame["Title"][labels == i]:
        print(' %s,' % title, end='')
    print() #add a whitespace

    print("Cluster %d words:" % i, end='')
    for ind in order_centroids[i, :no_words]:
        print (' %s' % terms[ind], end=','),
    print()
    print()