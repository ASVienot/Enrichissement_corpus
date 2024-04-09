from dataclasses import dataclass
from typing import List
import sys
import lxml
from bs4 import BeautifulSoup
import csv


@dataclass
class Seance:
    id_seance: str
    nb_tours_parole: int

@dataclass
class Token:
    form: str

@dataclass
class Phrase:
    info_phrase: str
    tokens: List[Token]
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
        #print(soup)
        liste_tours_parole = []
        i = 1
        for rda_tag in soup.find_all("rda"):
            id_rda = i
            i += 1
            sex = rda_tag['sex']
            nom = rda_tag['normalized_name']
            label = rda_tag['label']

            liste_phrases_par_rda = []
            for s_tag in rda_tag.find_all("s"):
                words = s_tag.find_all("w")
                texte_parties = []
                for w in words:
                    txm_form = w.find("txm:form")
                    if txm_form is not None and txm_form.text is not None:
                        texte_parties.append(txm_form.text)
                        token = Token(form=txm_form.text)
                        
                texte = ' '.join(texte_parties)
                id_phrase = s_tag.get('xml:id', "")
                info_phrase = (id_phrase, texte)
                phrase = Phrase(info_phrase=info_phrase, tokens=texte_parties)
                liste_phrases_par_rda.append(phrase)
                
            
            nb_phrases = len(liste_phrases_par_rda)
            tour_parole = ToursParole(id_rda=id_rda, sex=sex, nom=nom, label=label, nb_phrases=nb_phrases, phrases=liste_phrases_par_rda)
            liste_tours_parole.append(tour_parole)

              
    text_elem = soup.find('text')
    id_seance = text_elem.get('id', "")
    nb_tours_parole = len(liste_tours_parole)
    info_seance = Seance(id_seance=id_seance, nb_tours_parole=nb_tours_parole)


    print(f"id_seance = {info_seance.id_seance}, nb_tours_parole = {info_seance.nb_tours_parole}")
    for tour in liste_tours_parole:
        print(f"Id_tour : {tour.id_rda}, Personne: {tour.nom}, sexe: {tour.sex}, label: {tour.label}, nb_phrases: {tour.nb_phrases}, phrase: {[phrase.info_phrase for phrase in tour.phrases]}, tokens par phrase : {[phrase.tokens for phrase in tour.phrases]}" )
        print("-"*100)

    return liste_tours_parole        



##################################################################
"""
Need bug fix : Il faut pas utiliser chemin_fichier=sys.argv[1] car sinon on interprète "-h" comme un Path et non comme --help

def creation_tableau():
    liste_tours_parole = extraction_tours_parole(chemin_fichier=sys.argv[1])
    with open("exemple.csv", "w") as fichier:
        structure = csv.writer(fichier, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        structure.writerow(["Tour", "Id_phrase", "phrase"])
        for tour in liste_tours_parole:
            for phrase in tour.phrases:
                structure.writerow([tour.id_rda, phrase.info_phrase[0], phrase.info_phrase[1]])
                print(f"{tour.id_rda}\t{phrase.info_phrase[0]}\t{phrase.info_phrase[1]}")

creation_tableau()
"""
