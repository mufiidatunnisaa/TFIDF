#%%
import pandas as pd
import pprint
import nltk
import time
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

start = time.time()
# create stemmer
stemmer = PorterStemmer()

# create lemmatizer
lem = WordNetLemmatizer()

#stopword
stop_words = set(stopwords.words('english'))

df = pd.read_csv('textm_wine_reviews.csv', nrows=500, delimiter=',')
label_data = pd.DataFrame(df, columns=['winery','province','country'])
label = list(map(lambda x: ', '.join(x), label_data.values))

input_search = [
  ' memory of a wine once made by his mother'
]

list_document = list()
vectorizer = CountVectorizer()

for i in range(0, len(df)):
  tokenized_word=word_tokenize(df.loc[i][2])
  filter_words = [w for w in tokenized_word if not w in stop_words]
  stemmed_word = list(map(lambda x: stemmer.stem(x), filter_words))
  lemmatize_word = list(map(lambda x : lem.lemmatize(x, 'v'), stemmed_word))
  output = ' '.join(lemmatize_word)
  X = vectorizer.fit_transform([output])
  print(X.toarray())
  Y = vectorizer.transform(input_search)
  list_document.append([label[i], sum(Y.data)])

pprint.pprint(sorted(list_document, key = lambda x: x[1], reverse = True)[0:7])
end = time.time()
print('exe time Count Vectorizer: ', end-start)