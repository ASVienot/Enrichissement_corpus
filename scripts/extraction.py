from dataclasses import dataclass
from typing import List
import sys
import lxml
from bs4 import BeautifulSoup
import csv
from my_minipack.loading import ft_progress
"""Ce programme permet d'extraire une sutructure d'une séance, ce programme est appelé dans extraction_dossier.py"""

@dataclass
class Seance:
    id_seance: str
    nb_tours_parole: int

@dataclass
class Token:
    form: str
    lemme: str

@dataclass
class Phrase:
    info_phrase: str
    desaccord : bool
    tokens: List[Token]
    lemmes : List[Token]

@dataclass
class ToursParole:
    id_rda: int
    sex: str
    nom: str
    label: str
    nb_phrases: int
    phrases: List[Phrase]

def extraction_tours_parole(chemin_fichier):

    with open (chemin_fichier, 'r') as tei: 
        donnees = tei.read()
        soup = BeautifulSoup (donnees, 'lxml-xml')
        compteur_token = 0
        liste_tours_parole = []
        i = 1
        compteur_rda = 0

        for rda_tag in ft_progress(soup.find_all("rda")):
            id_rda = i
            i += 1
            sex = rda_tag['sex']
            nom = rda_tag['normalized_name']
            label = rda_tag['label']
            compteur_rda += 1

            liste_phrases_par_rda = []
            for s_tag in rda_tag.find_all("s"):
                words = s_tag.find_all("w")
                texte_parties = []
                liste_lemmes= []
                tok_desaccord = 0 

                for w in words:
                    compteur_token += 1
                    txm_form = w.find("txm:form")
                    txm_lemme = w.find("txm:ana", {"type":"#udlemma"}) 
                    ana_tag = w.find("txm:ana", {"resp":"#adm-sitri"})
                    if txm_form is not None and txm_form.text is not None:
                        texte_parties.append(txm_form.text)
                    if txm_lemme is not None and txm_lemme.text is not None:
                        liste_lemmes.append(txm_lemme.text)
                    token = Token(form=txm_form.text, lemme= txm_lemme.text if txm_lemme is not None else "")
                    if ana_tag is not None and ana_tag.text is not None and ana_tag.text == "D":
                        tok_desaccord += 1 
                    
                        
                desaccord = "D" if tok_desaccord > 0 else "A"        
                texte = ' '.join(texte_parties)
                lemmes = ' '.join(liste_lemmes)
                id_phrase = s_tag.get('xml:id', "")
                info_phrase = (id_phrase, texte, lemmes)
                phrase = Phrase(info_phrase=info_phrase, desaccord=desaccord, tokens=texte_parties, lemmes=liste_lemmes)
                liste_phrases_par_rda.append(phrase)
                
            
            nb_phrases = len(liste_phrases_par_rda)
            tour_parole = ToursParole(id_rda=id_rda, sex=sex, nom=nom, label=label, nb_phrases=nb_phrases, phrases=liste_phrases_par_rda)
            liste_tours_parole.append(tour_parole)

              
    text_elem = soup.find('text')
    id_seance = text_elem.get('id', "")
    nb_tours_parole = len(liste_tours_parole)
    info_seance = Seance(id_seance=id_seance, nb_tours_parole=nb_tours_parole)


    return liste_tours_parole        
