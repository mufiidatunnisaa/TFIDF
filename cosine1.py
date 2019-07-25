#%%
import pandas as pd
import pprint
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
import time
from nltk.tokenize import word_tokenize

start = time.time() 
# create stemmer
stemmer = PorterStemmer()

# create lemmatizer
lem = WordNetLemmatizer()

#stopword
stop_words = set(stopwords.words('english'))

read_data = pd.read_csv('textm_wine_reviews.csv', nrows=500, delimiter=',')
label_data = pd.DataFrame(read_data, columns=['winery','province','country'])
label = list(map(lambda x: ', '.join(x), label_data.values))


input_search = [
  'memory of a wine once made by his mother'
]

list_document = list()
vectorizer = CountVectorizer()
tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()

for i in range(0, len(read_data)):
  tokenized_word=word_tokenize(read_data.loc[i].description)
  filter_words = [w for w in tokenized_word if not w in stop_words]
  stemmed_word = list(map(lambda x: stemmer.stem(x), filter_words))
  lemmatize_word = list(map(lambda x : lem.lemmatize(x, 'v'), stemmed_word))
  output = ' '.join(lemmatize_word)
  X = vectorizer.fit_transform([output])
  Y = vectorizer.transform(input_search)
  cosine = cosine_similarity(X, Y)
  # print(cosine.toarray())
  list_document.append([label[i],cosine[0][0]])

pprint.pprint(sorted(list_document, key = lambda x: x[1], reverse = True)[0:7])
end = time.time()
print("Excecution Time cosine: ", end-start)
#%%