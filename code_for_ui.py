# will need these functions and imports for the uI code to clean user input 

# keys
# 0 - Fake 
# 1 - Real

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pickle import load

model = load(open("models/model.pkl",'rb'))
bow = load(open("models/bow.pkl",'rb'))

stopwords = set(stopwords.words('english'))

def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stopwords])

lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return " ".join(lemmatizer.lemmatize(word) for word in text.split())

def prep_headline(headline):
    headline = remove_stopwords(headline)
    headline = lemmatize_words(headline)    
    headline_bow = bow.transform([headline]).toarray()
    return headline_bow

# headline from today as an example 
headline = prep_headline('Whistleblower says Patrick Brown approved third-party payment amid Conservative campaign')

prediction = model.predict(headline)

predcition_prob = model.predict_proba(headline)

print(prediction, prediction_prob)

