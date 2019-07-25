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
start = time.time() 
# nltk.download('stopwords')
# nltk.download('wordnet')
def stem(token_normalised_text):

#stems a normalised tokens list.

	processed_text = []
	stemmer = PorterStemmer()

	for w in token_normalised_text:

		root = stemmer.stem(w)
		root = str(root)

		processed_text.append(root)

	return processed_text

def filtering(token_normalised_text):

#stems a normalised tokens list.

	processed_text = []
	stopWords = set(stopwords.words('english'))

	for w in token_normalised_text:
		if w not in stopWords:
			processed_text.append(w)

	return processed_text

read_data = pd.read_csv('textm_wine_reviews.csv', nrows=500, delimiter=',')
label_data = pd.DataFrame(read_data, columns=['winery','province','country'])
label = list(map(lambda x: ', '.join(x), label_data.values))
data = read_data.description
input_search = [
  'memory of a wine once made by his mother'
]

list_document = list()
vectorizer = CountVectorizer()
tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()

for i in range(0, len(read_data)):
  output = tokenizer.tokenize(str(data[i]))
  output = filtering(output)
  output = stem(output)
  output = ' '.join([lemmatizer.lemmatize(w,'v') for w in output])
  X = vectorizer.fit_transform([output])
  Y = vectorizer.transform(input_search)
  cosine = cosine_similarity(X, Y)
  list_document.append([label[i],cosine[0][0]])

pprint.pprint(sorted(list_document, key = lambda x: x[1], reverse = True)[0:7])
end = time.time()
print("Excecution Time cosine: ", end-start)
#%%
