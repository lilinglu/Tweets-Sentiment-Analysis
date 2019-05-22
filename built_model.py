import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from wordcloud import WordCloud, STOPWORDS

from textblob import TextBlob

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

LogisticRegression().get_params()

df = pd.read_csv('train.csv',encoding = "ISO-8859-1")

df.head()

df.Sentiment.unique()

df = df[:10000]

df.shape

def preprocess_text(text):
    # remove url
    remove_link = re.sub('(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})',"",text)
    # remove @mention
    text = re.sub('@[^\s]+','', text)
   
    #text = " ".join(re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])',"",text.lower()).split())
    #text = re.sub(' +',' ', text)
    # remove all the non letter characters including numbers
    text = re.sub("[^a-zA-Z]", " ", text)
    # reduce duplicated letters to 2
    text = re.sub(r'(.)\1+', r'\1\1',text)
    # remove single letter
    text = ' '.join( [w for w in text.lower().split() if len(w)>1] )
    return text.strip()

df['processed_Text'] = df.SentimentText.apply(preprocess_text)

df.head(20)

def wordcloud(tweets,col):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(background_color="white",stopwords=stopwords,random_state = 2016).generate(" ".join([i for i in tweets[col]]))
    plt.figure( figsize=(20,10), facecolor='k')
    plt.imshow(wordcloud)
wordcloud(df,'processed_Text')

# Baseline model

x_train, x_validation_test, y_train, y_validation_test = train_test_split(df['processed_Text'], df['Sentiment'], test_size = 0.2, random_state=42)

x_validation,x_test,y_validation,y_test=train_test_split(x_validation_test,y_validation_test,test_size=0.5,random_state=33)

def get_tweet_sentiment(text): 
        ''' 
        Utility function to classify sentiment of passed tweet 
        using textblob's sentiment method 
        '''
        # create TextBlob object of passed tweet text 
        analysis = TextBlob(text) 
        # set sentiment 
        if analysis.sentiment.polarity > 0: 
            return 1
        else: 
            return 0

y_val_pred = x_validation.apply(get_tweet_sentiment)

con_matx = confusion_matrix(y_validation, y_val_pred, labels=[1,0])

con_matx

con_matx = pd.DataFrame(con_matx, index = ['positive','negative'],\
                        columns = ['predicted_pos','pridicted_neg'])

con_matx

print('accuracy score: {0:.2f}%'.format(accuracy_score(y_validation,y_val_pred)*100))

print('Classification Report\n')
print(classification_report(y_validation,y_val_pred))

#TextBlob sentiment analysis yielded 62.57% accuracy on the validation set, which is not so good. So I am gonna built a model by myself, hopefully it will perform better.

#names = ['RandomForest','Gradientboost','LogisticRegression','MultinomialNB']
#
#clf = [RandomForestClassifier(n_estimators = 100),GradientBoostingClassifier(),LogisticRegression(), MultinomialNB()]
#
#clf_zip = zip(names,clf)
#
#from sklearn.pipeline import Pipeline
#
#results=[]
#for n,c in clf_zip:
#    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,2), min_df=2,stop_words='english')),
#                         ('tfidf', TfidfTransformer()),
#                         ('clf', c)])
#    tuned_parameters = {
#        'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],        
#        'tfidf__use_idf': (True, False),
#        'tfidf__norm': ('l1', 'l2'),
#        'clf__n_estimators':[50,100,150,200]
#        'clf_max_depth':[int(x) for x in np.linspace(10, 50, num = 5)]
#            }
#    clf = GridSearchCV(text_clf, tuned_parameters, scoring='f1_macro',cv=5)
#    clf.fit(x_train, y_train)
#    results.append((accuracy_score(y_validation,clf.predict(x_validation))))
#table = pd.DataFrame(columns=['accuracy'],index=names)
#table['accuracy']=results
#print(table)

text_clf1 = Pipeline([('vect', CountVectorizer(ngram_range=(1,2), min_df=2,stop_words='english')),
                     ('tfidf', TfidfTransformer()),
                     ('clf', RandomForestClassifier())])
tuned_parameters1 = {
    'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],        
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__n_estimators':[50,100,150,200]
    #'clf__max_depth':[int(x) for x in np.linspace(10, 50, num = 5)]
}
clf1 = GridSearchCV(text_clf1, tuned_parameters1, scoring='f1_macro',cv=5)
clf1.fit(x_train, y_train)
accuracy1=accuracy_score(y_validation,clf1.predict(x_validation))

text_clf2 = Pipeline([('vect', CountVectorizer(ngram_range=(1,2), min_df=2,stop_words='english')),
                         ('tfidf', TfidfTransformer()),
                         ('clf', GradientBoostingClassifier())])
tuned_parameters2 = {
            'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],        
            'tfidf__use_idf': (True, False),
            'tfidf__norm': ('l1', 'l2'),
            'clf__learning_rate':[0.15,0.1,0.05,0.005], 
            'clf__n_estimators':[50,100,150,200]
    }
clf2 = GridSearchCV(text_clf2, tuned_parameters2, scoring='f1_macro',cv=5)
    
clf2.fit(x_train, y_train)
#clf2.best_params_
accuracy2=accuracy_score(y_validation,clf2.predict(x_validation))

text_clf3 = Pipeline([('vect', CountVectorizer(ngram_range=(1,2), min_df=2,stop_words='english')),
                         ('tfidf', TfidfTransformer()),
                         ('clf', LogisticRegression())])
tuned_parameters3 = {
    'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],        
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),            
    'clf__penalty': ['l1', 'l2'],
    'clf__C': np.logspace(0, 4, 10)
}
clf3 = GridSearchCV(text_clf3, tuned_parameters3, scoring='f1_macro',cv=5)
    
clf3.fit(x_train, y_train)
accuracy3=accuracy_score(y_validation,clf3.predict(x_validation))

text_clf4 = Pipeline([('vect', CountVectorizer(ngram_range=(1,2), min_df=2,stop_words='english')),
                         ('tfidf', TfidfTransformer()),
                         ('clf',MultinomialNB())])
tuned_parameters4 = {
            'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],        
            'tfidf__use_idf': (True, False),
            'tfidf__norm': ('l1', 'l2'),
            'clf__alpha': [1, 1e-1, 1e-2]
    }
clf4 = GridSearchCV(text_clf4, tuned_parameters4, scoring='f1_macro',cv=5)
    
clf4.fit(x_train, y_train)
clf4.best_params_
accuracy4=accuracy_score(y_validation,clf4.predict(x_validation))


print(max([accuracy1,accuracy2,accuracy3,accuracy4]))