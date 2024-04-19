import pandas as pd
from dataclasses import dataclass
import csv

# class for phrases in .csv file
@dataclass
class Phrase_csv:
    Tour : int
    Id_phrase : str
    phrase : str
    lemmes : str
    accord : str


# Angry word lists
angry_words = []
with open('angry_words.txt', 'r') as file:
    for word in file:
        angry_words.append(word.strip())
print('Angry word list loaded.')

totaldata = pd.read_csv('totaldonnees.csv')
print('Csv with no angry phrases loaded.')
print('Be patient, next step will take about an hour.')

list_phrases = []

""" This is the 1000 phrases version
count = 0
for index, row in totaldata.iterrows():
    next(totaldata.iterrows())
    if count > 1000:
        break
    else:
        phrase = Phrase_csv(Tour=row['Tour'],
                        Id_phrase=row['Id_phrase'],
                        phrase=row['phrase'],
                        lemmes=row['lemmes'].lower(),
                        accord=row['accord'])
        list_phrases.append(phrase)
        count +=1
"""

# This is the fuck you computer version (whole corpus)
for index, row in totaldata.iterrows():
    phrase = Phrase_csv(Tour=row['Tour'],
                    Id_phrase=row['Id_phrase'],
                    phrase=row['phrase'],
                    lemmes=row['lemmes'].lower(),
                    accord=row['accord'])
    list_phrases.append(phrase)
print('Congrats if your computer survived! Corpus loaded.')

# Find more desaccord phrases
for phrase in list_phrases:
    if phrase.accord == 'D':
        continue
    else:
        for angry_word in angry_words:
            if angry_word in phrase.lemmes:
                phrase.accord='D'
                break


nb_angry_phrases = 0

for phrase in list_phrases:
    if phrase.accord == 'D':
        nb_angry_phrases += 1

print(f'Number of angry phrases: {nb_angry_phrases}')


with open("more_angry_phrases.csv", "w") as fichier:
    structure = csv.writer(fichier, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    structure.writerow(["Tour", "Id_phrase", "phrase", "lemmes", "accord"])
    for phrase in list_phrases:
        structure.writerow([phrase.Tour, phrase.Id_phrase, phrase.phrase, phrase.lemmes, phrase.accord])

print('New csv with more angry phrases created.')
