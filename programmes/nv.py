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

    corpus_dataframe = pd.read_csv('more_angry_phrases.csv', header=0, usecols=[2,4], names=['phrase','accord'])
    rows_d = corpus_dataframe[corpus_dataframe['accord'] == "D"]
    print(rows_d)
    corpus_dataframe = corpus_dataframe.dropna(subset=['phrase'])
    train_corpus, test_corpus=train_test_split(corpus_dataframe, test_size=0.1, random_state=42,stratify=corpus_dataframe['accord'])


    # Créer une pipeline
    model = make_pipeline(TfidfVectorizer(stop_words=french_stop_words), MultinomialNB())
    # On convertie les données textuelles en matrices globalement, chaque mot a un "score".
    # On choisie MultinomialNaiveBayes parce que c'est ce qu'on utilise généralement pour les classifications de texte. 
    # La pipeline permet de faire passer ce qui est vectorisé dans le modèle Multinomial.S

    # Permet d'entraîner le modèle
    model.fit(train_corpus['phrase'], train_corpus['accord'])
    # On "fit" les données d'entrainement et les données cibles.
    # On dit que ces données vectorisées sont associées à telle ou telle réponse.


    liste_vraies_valeurs = test_corpus['accord'].to_list()
    test_evaluation = model.predict(test_corpus['phrase'])
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

    print("Taux d'accuracy :", accuracy_score(test_corpus['accord'], test_evaluation))

    print("Classification Report:")
    print(classification_report(test_corpus['accord'], test_evaluation))

    return test_evaluation, test_corpus
    # 32328 lignes dans totaltest.csv
    # 190 erreurs
    # taux_accuracy = Taux d'accuracy : 0.9941227418955704 donc ça semble logique ?


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
matrice_de_confusion(test_evaluation, test_corpus)