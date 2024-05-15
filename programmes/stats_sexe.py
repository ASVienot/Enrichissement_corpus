import pandas as pd
import re


tableau = pd.read_csv('taux_sexe.csv') 


def dates(fichier):
    date = re.search(r'\d{4}-\d{2}-\d{2}', fichier)
    return date.group(0)

tableau['date'] = tableau['Id_phrase'].apply(dates)
tableau['date'] = pd.to_datetime(tableau['date'])
stats = tableau.groupby(
    [tableau['date'].dt.year, 'sexe']).agg(nombre_phrases=('phrase', 'count'), nombre_desaccord=("accord", lambda x: (x=="D").sum()))
    # Alors là mdr merci stackoverflow

print(stats)
