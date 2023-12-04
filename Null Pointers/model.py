import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Load the CSV file
df = pd.read_csv('final2.csv')

# Preprocess the data
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_function(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    text = ' '.join(words)
    return text

df['text'] = df['text'].apply(preprocess_function)

X = df['text']
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

def predict_sentiment(keyword):
    keyword_tfidf = vectorizer.transform([keyword])
    prediction = clf.predict(keyword_tfidf)
    return "Positive sentiment" if prediction == 1 else "Negative sentiment"
