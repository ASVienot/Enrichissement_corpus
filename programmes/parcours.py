from demo_scikitlearn_tableau import *
from pathlib import Path


def creation_tableau_parcours(dossier : Path):
    liste_fichiers = dossier.glob('./*.xml')
    liste_tours_parole = []
    for fichier in liste_fichiers:
        try:
            tours_parole = extraction_tours_parole(fichier)
            liste_tours_parole.extend(tours_parole)
        except:
            print(f'Erreur dans fichier {fichier}.')
    with open("exemple.csv", "w") as fichier:
        structure = csv.writer(fichier, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        structure.writerow(["Tour", "Id_phrase", "phrase"])
        for tour in liste_tours_parole:
            for phrase in tour.phrases:
                structure.writerow([tour.id_rda, phrase.info_phrase[0], phrase.info_phrase[1]])
                #print(f"{tour.id_rda}\t{phrase.info_phrase[0]}\t{phrase.info_phrase[1]}")

def main(dossier):
    creation_tableau_parcours(dossier)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Prendre tous les fichiers xml du corpus')
    parser.add_argument('dossier', type=Path, help='Ajouter un dossier dans lequel il y a des fichiers xml.')
    args = parser.parse_args()
    main(dossier=args.dossier)
