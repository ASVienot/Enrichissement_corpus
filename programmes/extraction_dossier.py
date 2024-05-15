from extraction import *
from my_minipack.loading import ft_progress
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

    with open("taux_sexe.csv", "w") as fichier:
        structure = csv.writer(fichier, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        structure.writerow(["Tour", "sexe", "Id_phrase", "phrase", "lemmes", "accord"])
        for tour in liste_tours_parole:
            for phrase in tour.phrases:
                structure.writerow([tour.id_rda, tour.sex, phrase.info_phrase[0], phrase.info_phrase[1], phrase.info_phrase[2], phrase.desaccord])
                print(f"{tour.id_rda}\t{tour.sex}\t{phrase.info_phrase[0]}\t{phrase.info_phrase[1]}\t{phrase.desaccord}\t {phrase.info_phrase[2]}")

def main(dossier):
    creation_tableau_parcours(dossier)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Prendre tous les fichiers xml du corpus')
    parser.add_argument('dossier', type=Path, help='Ajouter un dossier dans lequel il y a des fichiers xml.')
    args = parser.parse_args()
    main(dossier=args.dossier)