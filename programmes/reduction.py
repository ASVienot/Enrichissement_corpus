import pandas as pd
import argparse


def reduction(fichier):
    df= pd.read_csv(fichier)

    lignes_A=df[df['accord'].str.contains('A')].head(1000)
    lignes_D = df[df['accord'].str.contains('D')]

    plus_petit=pd.concat([lignes_A,lignes_D])
    plus_petit.to_csv('A_1000.csv', index=False)

def main():

    parser = argparse.ArgumentParser(description='Utilise le fichier qui a extrait tout le corpus')
    parser.add_argument('fichier', help='Ajouter un fichier csv.')
    args = parser.parse_args()
    reduction(args.fichier)


if __name__ == "__main__":
    main()