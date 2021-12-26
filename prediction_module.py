import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
import pickle


nltk.download('wordnet')
nltk.download('stopwords')
stop_words = stopwords.words("english")
stop_words.extend(stopwords.words("russian"))
stop_words = set(stop_words)
def tokenizing_text(dataset):
    dataset = dataset.lower()
    set_words   = word_tokenize(dataset)
    set_words = [word for word in set_words if word.isalpha()]
    set_words = [word for word in set_words if word not in stop_words]
    lematizer = WordNetLemmatizer()

    set_words = [lematizer.lemmatize(word) for word in set_words]

    return set_words


def preparing_dataset(data_prepared):
    data_prepared['token'] = data_prepared['Text'].apply(tokenizing_text)
    data_prepared['token_merged'] = data_prepared.token.apply(lambda x: ' '.join(x))

    return data_prepared

#train_data = pd.read_pickle('train_data.csv')
#test_data = pd.read_pickle('test_data.csv')
#print(f'Размер теста :{test_data.shape}')

#test_data = test_data.Text

#train_data = preparing_dataset(train_data)
#test_data = preparing_dataset(test_data)


#full_data = pd.concat([train_data.token_merged, test_data.token_merged])\
  #  .reset_index(drop=True)
#vectorizer = TfidfVectorizer(max_df=0.55, min_df=5)
#vectorizer.fit(full_data)

#X_train_tfidf = vectorizer.transform(train_data.token_merged)
#X_test_tfidf = vectorizer.transform(test_data.token_merged)

#y = train_data.Score

#params = {'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'],
       #   'class_weight': ['balanced', None],
        #   'warm_start': [True, False],
        #   'early_stopping': [True, False]}

#sgd_gs = GridSearchCV(SGDClassifier(), params, n_jobs=-1)
#sgd_gs.fit(X_train_tfidf, y)ы

#with open('model.pkl', 'wb') as file:
  #   pickle.dump(sgd_gs.best_estimator_, file)

#with open('vectorizer.pkl', 'wb') as file:
    # pickle.dump(vectorizer, file)