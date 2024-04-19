import numpy as np
import matplotlib.pyplot as plt # Pour les graphiques, on y est pas encore mdr
import seaborn as sns; set()
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import nltk # Obligée d'importer nltk pour obtenir des stopwords en français, il n'y en a pas chez scikit learn
from nltk.corpus import stopwords
nltk.download('stopwords')


# COMMANDE LES AMIS : python3 demo_scikit.py
# tout simple

def entrainement():

    french_stop_words = stopwords.words('french')

    train_dataframe = pd.read_csv('train_demo.csv', header=0, usecols=[2,3], names=['phrase','désaccord'])
    train_dataframe = train_dataframe.dropna(subset=['phrase'])


    # Créer une pipeline
    model = make_pipeline(TfidfVectorizer(stop_words=french_stop_words), MultinomialNB())
    # On convertie les données textuelles en matrices globalement, chaque mot a un "score".
    # On choisie MultinomialNaiveBayes parce que c'est ce qu'on utilise généralement pour les classifications de texte. 
    # La pipeline permet de faire passer ce qui est vectorisé dans le modèle Multinomial.S

    # Permet d'entraîner le modèle
    model.fit(train_dataframe['phrase'], train_dataframe['désaccord'])
    # On "fit" les données d'entrainement et les données cibles.
    # On dit que ces données vectorisées sont associées à telle ou telle réponse.


    # On récupère les données du fichier test hop hop hop ! 
    test_dataframe = pd.read_csv('test_demo.csv', header=0, usecols=[2,3], names=['phrase','désaccord'])
    test_dataframe = test_dataframe.dropna(subset=['phrase'])
    liste_vraies_valeurs = test_dataframe['désaccord'].to_list()
    test_evaluation = model.predict(test_dataframe['phrase'])
    # On lui donne le même genre de données que dans le train.
    # ET EN GROS ! le modèle s'entraine ici sur la colonne 'phrase'.
    # on obtient la liste des ses réponses

    e = 0
    for i,j in zip(liste_vraies_valeurs,test_evaluation):
        if i != j:
            print(f"Vraie valeur : {i} \t Prédiction modèle : {j} ERREUR")
            e += 1
        else:
            print(f"Vraie valeur : {i} \t Prédiction modèle : {j}")
    print(e)

    print("Taux d'accuracy :", accuracy_score(test_dataframe['désaccord'], test_evaluation))

    return test_evaluation, test_dataframe
    # 32328 lignes dans totaltest.csv
    # 190 erreurs
    # taux_accuracy = Taux d'accuracy : 0.9941227418955704 donc ça semble logique ?


# Création d'une matrice de confusion et d'une heatmap oh lala : 
def matrice_de_confusion(test_evaluation, test_dataframe):


    matrice = confusion_matrix(test_dataframe['désaccord'], test_evaluation)
    sns.heatmap(matrice, square=True, annot=True, fmt='d', cbar=False,  xticklabels=['Accord (A)', 'Désaccord (D)'], yticklabels=['Accord (A)', 'Désaccord (D)'])
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
matrice_de_confusion(test_evaluation, test_dataframe)