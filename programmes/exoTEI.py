
import lxml
from bs4 import BeautifulSoup
from prettytable import PrettyTable
table = PrettyTable()
table.field_names=["token", "annotation"]
tei_doc="CA-Nanterre-2001-01-22-corr-new.ana.tt.ud.dep.xml"
with open (tei_doc, 'r') as tei: 
    soup = BeautifulSoup (tei, 'lxml-xml')
    for tag in soup.find_all("w"):
        form = tag.find("txm:form").text
        ana = tag.find("txm:ana", {"type": "#ud-feats"}).text #changer le type pour obtenir n'importe quel type d'annotation
        table.add_row([form, ana])
print(table)