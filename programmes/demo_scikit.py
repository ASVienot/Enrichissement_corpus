import numpy as np
import matplotlib.pyplot as plt # Pour les graphiques, on y est pas encore mdr
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import nltk # Obligée d'importer nltk pour obtenir des stopwords en français, il n'y en a pas chez scikit learn
from nltk.corpus import stopwords
nltk.download('stopwords')


# COMMANDE LES AMIS : python3 demo_scikit.py
# tout simple


french_stop_words = stopwords.words('french')

train_dataframe = pd.read_csv('train.csv', header=0, usecols=[2,3], names=['phrase','désaccord'])

# Créer une pipeline
model = make_pipeline(TfidfVectorizer(stop_words=french_stop_words), MultinomialNB())
# On convertie les données textuelles en matrices globalement, chaque mot a un "score".
# On choisie MultinomialNaiveBayes parce que c'est ce qu'on utilise généralement pour les classifications de texte. 

# Permet d'entraîner le modèle
model.fit(train_dataframe['phrase'], train_dataframe['désaccord'])

# On récupère les données du fichier test hop hop hop ! 
test_dataframe = pd.read_csv('test.csv', header=0, usecols=[2,3], names=['phrase','désaccord'])
liste_vraies_valeurs = test_dataframe['désaccord'].to_list()
test_evaluation = model.predict(test_dataframe['phrase'])
# ET EN GROS ! le modèle s'entraine ici sur la colonne 'phrase'.
# on obtient la liste des ses réponses

e = 0
for i,j in zip(liste_vraies_valeurs,test_evaluation):
    if i != j:
        print(f"Vraie valeure : {i} \t Prédiction modèle : {j} ERREUR")
        e += 1
    else:
         print(f"Vraie valeure : {i} \t Prédiction modèle : {j}")
print(e)

print("Taux d'accuracy :", accuracy_score(test_dataframe['désaccord'], test_evaluation))


# 216 lignes dans test.csv
# 10 erreurs
# taux_accuracy = Taux d'accuracy : 0.9537037037037037 donc ça semble logique ?