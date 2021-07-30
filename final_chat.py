import nltk
import numpy as np
import json
import random


# downloading model to tokenize message
nltk.download('punkt')
# downloading stopwords- not want these words to take up space in database or processing time
nltk.download('stopwords')
# downloading wordnet, which contains all lemmas of english language
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

from nltk.stem import WordNetLemmatizer


#Lemmatization- the process of grouping together the different forms of a word to a single item.

def clean_corpus(corpus):
  # splitting words - lowering every word in text and handling punctuation
  corpus = [ doc.lower() for doc in corpus]
  cleaned_corpus = []
  
  stop_words = stopwords.words('english')
  wordnet_lemmatizer = WordNetLemmatizer()

  # iterating over every text
  # Iteration- process of looping through the objects ina collection
  for doc in corpus:
    # tokenizing text
    # simplifies the words
    tokens = word_tokenize(doc)
    cleaned_sentence = [] 
    for token in tokens: 
      # removing stopwords, and punctuation
      if token not in stop_words and token.isalpha(): 
        # applying lemmatization
        cleaned_sentence.append(wordnet_lemmatizer.lemmatize(token)) 
    cleaned_corpus.append(' '.join(cleaned_sentence))
  return cleaned_corpus


with open(r'C:\Users\Kanika\.vscode\extensions\ritwickdey.liveserver-5.6.1\node_modules\live-server\intents.json') as file:
  intents = json.load(file)

corpus = []
tags = []
for intent in intents['intents']:
    # taking all patterns in intents to train a neural network
    for pattern in intent['patterns']:
        corpus.append(pattern)
        tags.append(intent['tag'])

#sentences without the unnecessary words
cleaned_corpus = clean_corpus(corpus)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned_corpus)


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
y = encoder.fit_transform(np.array(tags).reshape(-1,1))

    
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([Dense(128, input_shape=(X.shape[1],), activation='relu'),
                    Dropout(0.2),
                    Dense(64, activation='relu'),
                    Dropout(0.2),
                    Dense(y.shape[1], activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(X.toarray(), y.toarray(), epochs=20, batch_size=1)

# if prediction for every tag is low, then we want to classify that message as noanswer
INTENT_NOT_FOUND_THRESHOLD = 0.40

def predict_intent_tag(message):
  message = clean_corpus([message])
  X_test = vectorizer.transform(message)
  y = model.predict(X_test.toarray())
  # if probability of all intent is low, classify it as noanswer
  if y.max() < INTENT_NOT_FOUND_THRESHOLD:
    return 'noanswer'
  
  prediction = np.zeros_like(y[0])
  prediction[y.argmax()] = 1
  tag = encoder.inverse_transform([prediction])[0][0]
  return tag

def get_intent(tag):
  # to return complete intent from intent tag
  for intent in intents['intents']:
    if intent['tag'] == tag:
      return intent


while True:
  # get message from user
  message = input('You: ')
  # predict intent tag using trained neural network
  tag = predict_intent_tag(message)
  # get complete intent from intent tag
  intent = get_intent(tag)
  # generate random response from intent
  response = random.choice(intent['responses'])
  print('Bot: ', response)

  # break loop if intent was goodbye
  if tag == 'goodbye':
    break
