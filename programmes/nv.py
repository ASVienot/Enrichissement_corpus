import csv
import numpy as np
import matplotlib.pyplot as plt # Pour les graphiques, on y est pas encore mdr
import seaborn as sns; set()
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import nltk # Obligée d'importer nltk pour obtenir des stopwords en français, il n'y en a pas chez scikit learn
from nltk.corpus import stopwords
nltk.download('stopwords')


# COMMANDE LES AMIS : python3 demo_scikit.py
# tout simple

def entrainement():


    french_stop_words = stopwords.words('french')
    french_stop_words.remove("pas")
    

    corpus_dataframe = pd.read_csv('data/corpus_annote.csv', header=0, usecols=[2,4], names=['phrase','accord'])
    corpus_dataframe = corpus_dataframe.dropna(subset=['phrase'])
    train_corpus, test_corpus=train_test_split(corpus_dataframe, test_size=0.25, random_state=42,stratify=corpus_dataframe['accord'])


    # Créer une pipeline
    model = make_pipeline(TfidfVectorizer(stop_words=french_stop_words), MultinomialNB())
    model.fit(train_corpus['phrase'], train_corpus['accord'])
    test_evaluation = model.predict(test_corpus['phrase'])
    # On convertie les données textuelles en matrices globalement, chaque mot a un "score".
    # On choisie MultinomialNaiveBayes parce que c'est ce qu'on utilise généralement pour les classifications de texte. 
    # La pipeline permet de faire passer ce qui est vectorisé dans le modèle Multinomial.S

    # Permet d'entraîner le modèle
    model.fit(train_corpus['phrase'], train_corpus['accord'])
    # On "fit" les données d'entrainement et les données cibles.
    # On dit que ces données vectorisées sont associées à telle ou telle réponse.


    test_evaluation = model.predict(test_corpus['phrase'])
    # On lui donne le même genre de données que dans le train.
    # ET EN GROS ! le modèle s'entraine ici sur la colonne 'phrase'.
    # on obtient la liste des ses réponses

    print("Taux d'accuracy :", accuracy_score(test_corpus['accord'], test_evaluation))
    print("Classification Report:")
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

    for phrase, vraie_valeur, prediction in zip(test_corpus["phrase"], test_corpus["accord"], test_evaluation):

        if vraie_valeur == 'D' and prediction == 'D':
            VP.append((phrase, vraie_valeur, prediction))
        elif vraie_valeur == 'A' and prediction == 'D':
            FP.append((phrase, vraie_valeur, prediction))
        elif vraie_valeur == 'A' and prediction == 'A':
            VN.append((phrase, vraie_valeur, prediction))
        elif vraie_valeur == 'D' and prediction == 'A':
           FN.append((phrase, vraie_valeur, prediction))

    with open("data_resultats/data_prediction_nv.csv", "w") as fichier:
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