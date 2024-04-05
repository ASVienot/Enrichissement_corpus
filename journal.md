## NOS ANNOTATIONS

### État des lieux

Nombre de fichiers : 50 comptes-rendus (1984-2002)

- RDA : tour de parole

# 15/03/2024

## Weiqi
### À faire :
- Ecrire un documentation qui explique la strcture du fichier et tous les attributs
- Connaître Scikit Learn


05/04 
# Alix 
Weiqi -> par


# Rapport 

## Présentation du corpus 
Ce corpus: 
- créer par les archives de l'université Paris Nanterre
- regroupe les archives du conseil d'administration de l'université depuis 1984 jusqu'en 2018
Ce corpus a été annoté par ...
L'annotation est divisé par tours de paroles qui sont annotés avec le locuteur son genre et la fonction qu'il exerce dans l'université. Elle est ensuite divisé par phrases puis par tokens. 
Les balises des tokens contiennent des informations sur les tokens individuels : leur pos, leur lemme, leur features, relation de dépendance, leur tête avec sa forme, lemme et son pos, id de la relation de dépendance et sa forme et son schéma.
Une dernière balise contient l'information sur lequel repose notre projet : le désaccord
Elle est de forme : <txm:ana type="#valeurvdi" resp="#adm-sitri">D</txm:ana>, et peut prendre la valeur D exprimant le désaccord, INDEF ou vide.

Dans le corpus complet, on trouve 378 documents pour un total de X tours de paroles, X phrases et X tokens. 
Pour les annotations, on trouve X balises "#adm-sitri" dont X sont de valeur "D". 


## Scripts de récupération des données
Maria : extrait l'information contenu dans les balise "#adm-sitri" et les mets dans un format TEI. 
Pour la création du tableau, le script ouvre un dossier et parcours tous les fichiers en récupérant les données nécesssaires à la création du tableau. 
On trouve dans le tableau, l'id du tour de parole, l'id de la phrase, la phrase dont les tokens ont été concaténé et en dernier la valeur A ou D selon si la phrase contient des tokens avec la balise de désaccord. 
On peut rajouter à ce tableau la liste des lemmes afin d'entrainer la machine sur des schémas syntaxiques similaires. 


## Script Scikit Learn 
Nous avons choisi d'entrainer la machine sur X % du corpus. 

## Résultats 

## Conclusion 
Le corpus est nul et non avenu.




