import csv
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns; set()
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import nltk # Obligée d'importer nltk pour obtenir des stopwords en français, il n'y en a pas chez scikit learn
from nltk.corpus import stopwords
nltk.download('stopwords')

# python3 svm.py

"""
Ce programme permet d'entrainer un modèle SVM sur notre corpus d'origine et complet (totaldonnees.py) mais peut être aussi utilisé avec les corpus modifiés (corpus_annote.csv et A_1000.csv).
"""

def entrainement():


    french_stop_words = stopwords.words('french')
    french_stop_words.remove("pas")
    

    corpus_dataframe = pd.read_csv('../data/totaldonnees.csv', header=0, usecols=[2,4], names=['phrase','accord'])
    corpus_dataframe = corpus_dataframe.dropna(subset=['phrase'])
    train_corpus, test_corpus=train_test_split(corpus_dataframe, test_size=0.25, random_state=42,stratify=corpus_dataframe['accord'])

     # Vectorisation TF-IDF
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('french'), max_df=0.03)
    X_train_tfidf = vectorizer.fit_transform(train_corpus['phrase'])

    # Extraction des termes et leurs scores TF-IDF moyens
    feature_names = vectorizer.get_feature_names_out()
    scores = X_train_tfidf.mean(axis=0).tolist()[0]  # Calcul des scores moyens
    term_scores = dict(zip(feature_names, scores))

    sorted_term_scores = sorted(term_scores.items(), key=lambda x: x[1])

    # Affichage des résultats
    print("Termes et leurs scores TF-IDF en ordre croissant :")
    for term, score in sorted_term_scores:
        print(f"{term}: {score}")

    # Créer une pipeline
    model = make_pipeline(TfidfVectorizer(stop_words=french_stop_words), SVC())
    model.fit(train_corpus['phrase'], train_corpus['accord'])
    test_evaluation = model.predict(test_corpus['phrase'])
    # On convertie les données textuelles en matrices globalement, chaque mot a un "score".
    # Permet d'entraîner le modèle
    # On "fit" les données d'entrainement et les données cibles.
    # On dit que ces données vectorisées sont associées à telle ou telle réponse.


    X_train_tfidf = vectorizer.transform(train_corpus['phrase'])
    print("Importance de chaque feature pour l'entrainement :")
    print(X_train_tfidf)


    print("Taux d'accuracy :", accuracy_score(test_corpus['accord'], test_evaluation))
    print("Classification Report:")
    print(classification_report(test_corpus['accord'], test_evaluation))

    return test_evaluation, test_corpus

def obtention_phrases(test_evaluation, test_corpus):

    VP = []
    FP = []
    VN = []
    FN = []

    for phrase, vraie_valeur, prediction in zip(test_corpus["phrase"], test_corpus["accord"], test_evaluation):

        if vraie_valeur == 'D' and prediction == 'D':
            VP.append((phrase, vraie_valeur, prediction))
        elif vraie_valeur == 'A' and prediction == 'D':
            FP.append((phrase, vraie_valeur, prediction))
        elif vraie_valeur == 'A' and prediction == 'A':
            VN.append((phrase, vraie_valeur, prediction))
        elif vraie_valeur == 'D' and prediction == 'A':
           FN.append((phrase, vraie_valeur, prediction))

    with open("data_resultats/data_prediction_svm.csv", "w") as fichier:
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
def matrice_de_confusion(test_evaluation, test_corpus):


    matrice = confusion_matrix(test_corpus['accord'], test_evaluation)
    sns.heatmap(matrice, square=True, annot=True, fmt='d',cmap='Spectral', cbar=False, xticklabels=['Accord (A)', 'Désaccord (D)'], yticklabels=['Accord (A)', 'Désaccord (D)'])
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
obtention_phrases(test_evaluation, test_corpus)
matrice_de_confusion(test_evaluation, test_corpus)