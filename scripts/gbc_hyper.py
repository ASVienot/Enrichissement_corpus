import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; set()
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import nltk # Obligée d'importer nltk pour obtenir des stopwords en français, il n'y en a pas chez scikit learn
from nltk.corpus import stopwords
nltk.download('stopwords')


"""
Ce programme permet d'entrainer un modèle GBC avec modification d'hyperparamètres sur notre corpus d'origine et complet (totaldonnees.py) mais peut être aussi utilisé avec les corpus modifiés (corpus_annote.csv et A_1000.csv).
"""
# COMMANDE LES AMIS : python3 gbc_hyper.py
# tout simple

def entrainement():

    french_stop_words = stopwords.words('french')
    french_stop_words.remove("pas")
    

    corpus_dataframe = pd.read_csv('data/totaldonnees.csv', header=0, usecols=[2,4], names=['phrase','accord'])
    corpus_dataframe = corpus_dataframe.dropna(subset=['phrase'])
    train_corpus, test_corpus=train_test_split(corpus_dataframe, test_size=0.25, random_state=42,stratify=corpus_dataframe['accord'])
    
    parameters = {
        #limite de la fréquence  maximal de documents dans laquelle on va permettre un terme
        'tfidfvectorizer__max_df': (0.5, 0.7),
        #quantité maximale de caractéristiques qu'on va utiliser pour la vectorisation
        'tfidfvectorizer__max_features': (None, 10000),
        #n-gramme, unigramme
        'tfidfvectorizer__ngram_range': ((1, 1), (1, 2)),  
        'gradientboostingclassifier__n_estimators': [150],
        'gradientboostingclassifier__max_depth': [3],
        'gradientboostingclassifier__learning_rate': [0.1]
    }

    # Créer une pipeline
    pipeline = make_pipeline(TfidfVectorizer(stop_words=french_stop_words), 
                          GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1))
    
    # On convertie les données textuelles en matrices globalement, chaque mot a un "score".
 
    grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(train_corpus['phrase'], train_corpus['accord'])
    #J'obtiens le meilleur modèle 
    best_model = grid_search.best_estimator_

    test_evaluation = best_model.predict(test_corpus['phrase'])
    # On "fit" les données d'entrainement et les données cibles.
    # On dit que ces données vectorisées sont associées à telle ou telle réponse.

    liste_vraies_valeurs = test_corpus['accord'].to_list()
    
    # On lui donne le même genre de données que dans le train.
    # ET EN GROS ! le modèle s'entraine ici sur la colonne 'phrase'.
    # on obtient la liste des ses réponses

    print("Taux d'accuracy :", accuracy_score(test_corpus['accord'], test_evaluation))

    print("Classification Report:")
    print(f"Voici les meilleurs paramètres:{grid_search.best_params_}")
    print(classification_report(test_corpus['accord'], test_evaluation))

    return test_evaluation, test_corpus
    # 32328 lignes dans totaltest.csv
    # 190 erreurs
    # taux_accuracy = Taux d'accuracy : 0.9941227418955704 donc ça semble logique ?

def obtention_phrases(test_evaluation, test_corpus):

    VP = []
    FP = []
    VN = []
    FN = []
    print("on arrive la ?")
    for phrase, vraie_valeur, prediction in zip(test_corpus["phrase"], test_corpus["accord"], test_evaluation):

        if vraie_valeur == 'D' and prediction == 'D':
            VP.append((phrase, vraie_valeur, prediction))
        elif vraie_valeur == 'A' and prediction == 'D':
            FP.append((phrase, vraie_valeur, prediction))
        elif vraie_valeur == 'A' and prediction == 'A':
            VN.append((phrase, vraie_valeur, prediction))
        elif vraie_valeur == 'D' and prediction == 'A':
           FN.append((phrase, vraie_valeur, prediction))

    with open("data_resultats/data_prediction_gbc_hyper.csv", "w") as fichier:
        structure = csv.writer(fichier, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        structure.writerow(["phrase", "valeur réelle", "valeur prédite"])
        if len(VP) > 0:
            structure.writerow(["VRAIS POSITIFS", " ", " "])
        for phrase in VP:
            structure.writerow([phrase[0], phrase[1], phrase[2]])
        if len(FP) > 0:
            structure.writerow(["FAUX POSITIFS", " ", " "])
        for phrase in FP:
            structure.writerow([phrase[0], phrase[1], phrase[2]])
        if len(VN) > 0:
            structure.writerow(["VRAIS NÉGATIFS", " ", " "])
        for phrase in VN:
            structure.writerow([phrase[0], phrase[1], phrase[2]])
        if len(FN):
            structure.writerow(["FAUX NÉGATIFS", " ", " "])
        for phrase in FN:
            structure.writerow([phrase[0], phrase[1], phrase[2]])
        

# Création d'une matrice de confusion et d'une heatmap oh lala : 
def matrice_de_confusion(test_evaluation, test_dataframe):


    matrice = confusion_matrix(test_dataframe['accord'], test_evaluation)
    sns.heatmap(matrice, square=True, annot=True, cmap='Spectral', fmt='d', cbar=False, xticklabels=['Accord (A)', 'Désaccord (D)'], yticklabels=['Accord (A)', 'Désaccord (D)'])
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




test_evaluation, test_dataframe = entrainement()
obtention_phrases(test_evaluation, test_dataframe)
matrice_de_confusion(test_evaluation, test_dataframe)