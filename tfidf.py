import pandas as pd
import pprint
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import time
start = time.time()

lemmatizer = WordNetLemmatizer()
# kata bantu
stop_words = set(stopwords.words('english'))
# create stemmer
stemmer = PorterStemmer()

df = pd.read_csv('textm_wine_reviews.csv', nrows=500, delimiter=',')

# Create Label
label_df = pd.DataFrame(df, columns=['winery', 'province', 'country'])
label = list(map(lambda x: ', '.join(x), label_df.values))

input_search = [
  'memory of a wine once made by his mother'
]

list_document = list()
vectorizer = TfidfVectorizer()

for i in range(0, len(df)):
  tokenized_word = word_tokenize(df.loc[i].description)
  filtered_word = [w for w in tokenized_word if not w in stop_words]
  stemmed_word = list(map(lambda x: stemmer.stem(x), filtered_word))
  lemmatized_word = list(map(lambda x: lemmatizer.lemmatize(x, 'v'), stemmed_word))
  output = ' '.join(lemmatized_word)
  X = vectorizer.fit_transform([output])
  # print(X.toarray())
  Y = vectorizer.transform(input_search)
  # print(Y.data)
  list_document.append([label[i], sum(Y.data)])

pprint.pprint(sorted(list_document, key = lambda x: x[1], reverse = True)[0:7])
end = time.time()
print('Execution Time Tfidf: ', end-start)