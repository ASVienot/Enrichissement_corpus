import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt # Pour les graphiques, on y est pas encore mdr
import seaborn as sns; set()
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import nltk # Obligée d'importer nltk pour obtenir des stopwords en français, il n'y en a pas chez scikit learn
from nltk.corpus import stopwords
nltk.download('stopwords')


# COMMANDE LES AMIS : python3 demo_scikit.py
# tout simple

def entrainement(regime, index):


    french_stop_words = stopwords.words('french')
    french_stop_words.remove("pas")
    

    corpus_dataframe = pd.read_csv('recettes.csv', header=0, usecols=[2,index], names=['textes', regime])
    corpus_dataframe = corpus_dataframe.dropna(subset=['textes'])
    train_corpus, test_corpus=train_test_split(corpus_dataframe, test_size=0.25, random_state=42,stratify=corpus_dataframe[regime])


    # Créer une pipeline
    model = make_pipeline(TfidfVectorizer(stop_words=french_stop_words), KNeighborsClassifier())
    model.fit(train_corpus['textes'], train_corpus[regime])
    test_evaluation = model.predict(test_corpus['textes'])
    # On convertie les données textuelles en matrices globalement, chaque mot a un "score".
    # On choisie MultinomialNaiveBayes parce que c'est ce qu'on utilise généralement pour les classifications de texte. 
    # La pipeline permet de faire passer ce qui est vectorisé dans le modèle Multinomial.S

    # Permet d'entraîner le modèle
    # On "fit" les données d'entrainement et les données cibles.
    # On dit que ces données vectorisées sont associées à telle ou telle réponse.
    # On lui donne le même genre de données que dans le train.
    # ET EN GROS ! le modèle s'entraine ici sur la colonne 'phrase'.
    # on obtient la liste des ses réponses

    print("Taux d'accuracy :", accuracy_score(test_corpus[regime], test_evaluation))
    print("Classification Report:")
    print(classification_report(test_corpus[regime], test_evaluation))

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
def matrice_de_confusion(test_evaluation, test_corpus, regime):


    matrice = confusion_matrix(test_corpus[regime], test_evaluation)
    sns.heatmap(matrice, square=True, annot=True, fmt='d',cmap='Spectral', cbar=False, xticklabels=[f'non-{regime}', f'{regime}'], yticklabels=[f'non-{regime}',  f'{regime}'])
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

def main():
    
    parser = argparse.ArgumentParser(description="Choix du régime alimentaire pour l'entraînement")
    parser.add_argument('regime', choices=['vegetarien', 'vegan', 'crudivore', 'sans_gluten', 'alixproof', 'sans_noix', "sucré/salé"], help= "Choix du régime alimentaire")
    args = parser.parse_args()
        
    if args.regime == "vegetarien":
        regime = "vegetarien"
        index = 4
    elif args.regime =="vegan":
        regime = "vegan"
        index = 5
    elif args.regime == "crudivore":
        regime = "crudivore"
        index = 6
    elif args.regime == "sans_gluten":
        regime = "sans_gluten"
        index = 7
    elif args.regime == "alixproof": 
        regime = "alixproof"
        index = 8
    elif args.regime == "sans_noix":
        regime = "sans_noix"
        index = 9
    elif args.regime == "sucré/salé": 
        index = 10
    test_evaluation, test_corpus = entrainement(regime, index)
    #obtention_phrases(test_evaluation, test_corpus)
    matrice_de_confusion(test_evaluation, test_corpus, args.regime)


if __name__ == "__main__":
    main()