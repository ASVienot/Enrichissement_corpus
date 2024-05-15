import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns; set()
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
import nltk 
from nltk.corpus import stopwords
nltk.download('stopwords')


def entrainement():

    french_stop_words = stopwords.words('french')

    corpus_dataframe = pd.read_csv('corpus_total.csv', header=0, usecols=[2,4], names=['phrase','accord'])
    corpus_dataframe = corpus_dataframe.dropna(subset=['phrase'])
    train_corpus, test_corpus=train_test_split(corpus_dataframe, test_size=0.4, random_state=42,stratify=corpus_dataframe['accord'])
    #Cette partie on peut la modifier manuellement 
    hyper_parameters = {
        #limite de la fréquence  maximale de documents dans laquelle on va permettre un terme
        'tfidfvectorizer__max_df': (0.1, 0.2, 0.4, 0.8),
        #quantité maximale de caractéristiques qu'on va utiliser pour la vectorisation
        'tfidfvectorizer__max_features': (5000, 10000, 50000, 80000),
        #n-gramme, unigramme
        'tfidfvectorizer__ngram_range': ((1, 1), (1, 2), (1,3)),  
        'multinomialnb__alpha': (1e-2, 1e-3, 1e-1)}
    #Je crée le pipeline avec NB
    pipeline = make_pipeline(TfidfVectorizer(stop_words=french_stop_words), MultinomialNB())
    #Je cherche les hyperparamètres : cv: division croisée, n_jobs: il va utiliser tous les processeurs possibles.
    grid_search = GridSearchCV(pipeline, hyper_parameters, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(train_corpus['phrase'], train_corpus['accord'])
    #J'obtiens le meilleur modèle 
    best_model = grid_search.best_estimator_


    #Nous testons a partir de la division "test corpus"
    test_evaluation=best_model.predict(test_corpus['phrase'])
    print(f"Voici les meilleurs hyperparamètres:{grid_search.best_params_}")
    print("Taux d'accuracy :", accuracy_score(test_corpus['accord'], test_evaluation))

    return test_evaluation, test_corpus
    

def matrice_de_confusion(test_evaluation, test_corpus):


    matrice = confusion_matrix(test_corpus['accord'], test_evaluation)
    sns.heatmap(matrice, square=True, annot=True, fmt='d', cbar=False,  cmap="Spectral", xticklabels=['Accord (A)', 'Désaccord (D)'], yticklabels=['Accord (A)', 'Désaccord (D)'])
    plt.xlabel('Vraies valeurs')
    plt.ylabel('Valeurs prédites')
    plt.title('Matrice de confusion')
    plt.show()



test_evaluation, test_corpus = entrainement()
matrice_de_confusion(test_evaluation, test_corpus)