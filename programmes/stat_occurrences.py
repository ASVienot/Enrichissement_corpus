
### commande : python3 stat_occurrences.py --fichier data_prediction_gbc.csv --algo gbc

## Ce programme permet d'obtenir les occurrences de mots et de n-grams les plus communs dans les
## classes VP, FP, VN et FN de chaque test par algorithme. Les graphiques correspondant sont obtenus à l'aide de ce programme.
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
import nltk # Obligée d'importer nltk pour obtenir des stopwords en français, il n'y en a pas chez scikit learn
from nltk.corpus import stopwords
nltk.download('stopwords')
import matplotlib.pyplot as plt
import argparse  

def categorisation(tableau: Path):

    df = pd.read_csv(tableau, header=None, skiprows=1)
    
    categories = {}
    categorie = None

    for index, ligne in df.iterrows():
        if ligne[0] in ["VRAIS POSITIFS", "FAUX POSITIFS", "VRAIS NÉGATIFS", "FAUX NÉGATIFS"]:
            categorie = ligne[0]
            categories[categorie] = []
        elif categorie:
            categories[categorie].append(ligne[0])
    
    VP = [valeur for cle, valeur in categories.items() if cle == "VRAIS POSITIFS" ]
    FP = [valeur for cle, valeur in categories.items() if cle == "FAUX POSITIFS" ]
    VN = [valeur for cle, valeur in categories.items() if cle == "VRAIS NÉGATIFS" ]
    FN = [valeur for cle, valeur in categories.items() if cle == "FAUX NÉGATIFS" ]
    
    return VP, FP, VN, FN


def top_expressions(categorie, num):

    texte = [" ".join(phrase) for phrase in categorie] # On doit transformer notre texte en un élément d'une liste
    french_stop_words = stopwords.words('french')
    vectorisation = CountVectorizer(ngram_range = (num, num), max_features = 100, stop_words=french_stop_words)
    # Donc on vectorise tout notre texte et on donne des scores à chaque mot, nous établissons la taille des n-grams 
    sac_de_mots = vectorisation.fit_transform(texte) # On "apprend" le vocabulaire 
    sum_words = sac_de_mots.sum(axis = 0) 
    words_freq = [(word, sum_words[0, i]) for word, i in vectorisation.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True) # On obtient fréquences
    print(words_freq[:10]) 
    return words_freq[:10]

def representation_graphique(plus_frequents, nommage, categorie, ngram):

    nom = []
    taille = []

    for gram in plus_frequents:
        nom.append(gram[0])
        taille.append(gram[1])

    plt.figure(figsize=(14, 12))  
    plt.pie(taille, labels=nom)
    plt.title(f"Top {ngram}-grams pour {categorie} {nommage}")
    plt.axis('equal')
    plt.savefig(f"graphs/{nommage}_{categorie}_{ngram}.png")
    plt.show()
    plt.close()
 
def main():

    parser = argparse.ArgumentParser(description='Obtenir les pie charts de chaque catégorie de nos tests.')
    parser.add_argument('--fichier', type=Path, help='Nom du fichier csv correspondant')
    parser.add_argument('--algo', type=Path, help='Nom pour differencier les fichiers')
    args = parser.parse_args()
   
    taille = [1, 3]
    VP, FP, VN, FN = categorisation(tableau = args.fichier)
    liste_categories = [VP, FP, VN, FN]
    liste_categories_noms = ["VP", "FP", "VN", "FN"]
    for categorie, categorie_nom in zip(liste_categories, liste_categories_noms):
        for num in taille:
            plus_frequents = top_expressions(categorie, num)
            print(plus_frequents)
            representation_graphique(plus_frequents, nommage = args.algo, categorie=categorie_nom, ngram=num)

if __name__ == "__main__":
    main()