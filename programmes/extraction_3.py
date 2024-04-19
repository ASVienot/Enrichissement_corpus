from bs4 import BeautifulSoup
from pathlib import Path
from prettytable import PrettyTable
import lxml
table = PrettyTable()
table.field_names=["token", "annotation"]

def get_filenames(chemin_dossier: str) -> list[Path]:
    dossier = Path(chemin_dossier)
    return list(dossier.glob('**/*.xml'))

def analyse_bs(corpus):
    folder_list = get_filenames(corpus)
    for file_path in folder_list:
        with open(file_path, 'r') as tei: 
            soup = BeautifulSoup (tei, 'lxml-xml')
            for tag in soup.find_all("w"):
                ana_tag = tag.find("txm:ana", {"resp":"#adm-sitri"}) #changer le type pour obtenir n'importe quel type d'annotation
                
                form = tag.find("txm:form").text
                ana = ana_tag.text if ana_tag else 'Sans contenu'
                table.add_row([form, ana])
    print(table)
    
    with open('test.csv', 'w', newline='') as f_output:
        f_output.write(table.get_csv_string())
    
x = "./Corpus_hors"
analyse_bs(x)