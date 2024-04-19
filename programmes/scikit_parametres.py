import numpy as np
import matplotlib.pyplot as plt # Pour les graphiques, on y est pas encore mdr
import seaborn as sns; set()
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
import nltk # Obligée d'importer nltk pour obtenir des stopwords en français, il n'y en a pas chez scikit learn
from nltk.corpus import stopwords
nltk.download('stopwords')


# COMMANDE LES AMIS : python3 demo_scikit.py
# tout simple

def entrainement():

    french_stop_words = stopwords.words('french')

    corpus_dataframe = pd.read_csv('corpus_total.csv', header=0, usecols=[2,4], names=['phrase','accord'])
    corpus_dataframe = corpus_dataframe.dropna(subset=['phrase'])
    train_corpus, test_corpus=train_test_split(corpus_dataframe, test_size=0.4, random_state=42,stratify=corpus_dataframe['accord'])
    #Cette partie je dois la modifier manuellement
    parameters = {
        #limite de la fréquence  maximal de documents dans laquelle on va permettre un terme
        'tfidfvectorizer__max_df': (0.2, 0.4, 0.8),
        #quantité maximale de caractéristiques qu'on va utiliser pour la vectorisation
        'tfidfvectorizer__max_features': (None, 5000, 10000, 50000),
        #n-gramme, unigramme
        'tfidfvectorizer__ngram_range': ((1, 1), (1, 2)),  
        'multinomialnb__alpha': (1e-2, 1e-3)
    }
    #Je crée le pipeline avec NB
    pipeline = make_pipeline(TfidfVectorizer(stop_words=french_stop_words), MultinomialNB())
    #Je cherche les hyperparamètres : cv: division croisée, n_jobs: il va utiliser tous les processeurs possibles.
    grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(train_corpus['phrase'], train_corpus['accord'])
    #J'obtiens le meilleur modèle 
    best_model = grid_search.best_estimator_


    #Nous testons a partir de la division "test corpus"
    test_evaluation=best_model.predict(test_corpus['phrase'])
    print(f"Voici les meilleurs paramètres:{grid_search.best_params_}")
    print("Taux d'accuracy :", accuracy_score(test_corpus['accord'], test_evaluation))

    return test_evaluation, test_corpus
    

# Création d'une matrice de confusion et d'une heatmap oh lala : 
def matrice_de_confusion(test_evaluation, test_corpus):


    matrice = confusion_matrix(test_corpus['accord'], test_evaluation)
    sns.heatmap(matrice, square=True, annot=True, fmt='d', cbar=False,  cmap="Spectral", xticklabels=['Accord (A)', 'Désaccord (D)'], yticklabels=['Accord (A)', 'Désaccord (D)'])
    plt.xlabel('Vraies valeurs')
    plt.ylabel('Valeurs prédites')
    plt.title('Matrice de confusion')
    plt.show()

    # Dans cette matrice on a les vraies positives en bas à droite (plutôt):
        # Ce qui est vraiment D et qui a été reconnu comme tel.
    # Les faux positifs en haut à droite : 
        # Ce qui est compté comme D alors que A.
    # Les vrai négatifs en haut à gauche : 
        # Ce qui est A (donc négatifs) et bien compté comme A.
    # Les faux négatifs en bas à gauche :
        # Ce qui est est compté comme A alors que D.


test_evaluation, test_corpus = entrainement()
matrice_de_confusion(test_evaluation, test_corpus)