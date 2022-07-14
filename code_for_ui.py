# will need these functions and imports for the uI code to clean user input 

# keys
# 0 - Fake 
# 1 - Real

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pickle import load
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

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

st.title("Team 11: Fake News Detector")
st.text_input("Headline", key="input")
user_headline = st.session_state.input


# Use as test headlines:
# I had to dig through some vile online news sites to get those fake stories

# Very fake: Pro-Abort Berkeley Prof Accuses Sen. Hawley of ‘Violence’ for Saying Only Women Can Get Pregnant
# Very true: Trudeau defends decision to return Russia-owned turbine

# Fake: Sweden Rocked by Attempted Murder, Brutal Rape of Child at Playground by African Suspect
# True: Biden makes Holocaust gaffe during Israel visit

# Somewhat fake: Jan. 6 Committee Targets Alex Jones, Falsely Claims He Encouraged Violence At Capitol
# Somewhat true: Putin Pal Drops Menacing Hint: a ‘Cleansing’ Is Coming for Europe

# Unsure: Insurrectionist Anti-Vax Doctor Simone Gold Was Just Sentenced to 60 Days in Prison

if len(user_headline) != 0 : # if user has provided a headline, run the model and output results
    headline = prep_headline(user_headline)
    prediction = model.predict(headline)
    pred_output = "real" if prediction == 1 else "fake"
    prediction_prob = model.predict_proba(headline)
    prediction_prob = prediction_prob[0][prediction][0] # get the confidence of the prediction, which will be between 0.5 (least certain) and 1 (most certain)
    prediction_prob = round(prediction_prob,2)

    if prediction_prob < 0.6 :
        st.write("We are not sure whether this headline is real or fake. Be aware when reading this article.")
    elif prediction_prob < 0.7 :
        st.write("We are somewhat confident that this headline is " + pred_output + " (Confidence: " + str(prediction_prob) + " out of 1)")
    elif prediction_prob < 0.9 :
        st.write("We are confident that this headline is " + pred_output + " (Confidence: " + str(prediction_prob) + " out of 1)")
    else :
        st.write("We are very confident that this headline is " + pred_output + " (Confidence: " + str(prediction_prob) + " out of 1)")
else :
    st.write("Please input a news article headline, and our detector will attempt to determine whether it is real or fake.")

