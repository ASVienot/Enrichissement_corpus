# Enrichissement_corpus
## Projet Corpus PV de CA

## Instructions :

Archives de l’université Paris Nanterre : prédiction automatique de désaccord
- Données :
o Corpus des archives annotés en POS, lemmes, dépendances ? (à voir si besoin), infos sur gouverneur et dépendant, désaccords
- Contact :
o Dumoulin Hugo : hdumoulin@parisnanterre.fr
o Frédérique Sitri : Frederique.Sitri@u-pec.fr
- Objectif :
o Prédire automatiquement (avec l’apprentissage de surface Weka, Scikit Learn) tour de parole (lié à locuteur, balise RDA), phrase (signe de ponctuation à reconstituer avec un parseur), par token ? la présence de désaccord (oui/non)
 Se familiariser avec les données et les conventions (petit document expliquant les principes de l’annotation manuelle réalisée)
 Tour de parole et phrase : text classification, token : token classification ?
 Réfléchir sur les propriétés linguistique (features) à ajouter
 Test, entraînement, évaluation / validation croisée

##Outils 
__TXM-TEI__: 
 https://groupes.renater.fr/wiki/txm-info/public/xml_tei_txm
 _Pour la compréhension des annotations faites dans le corpus_
