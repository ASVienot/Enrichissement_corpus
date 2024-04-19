import pandas as pd

df = pd.read_csv('test_complet.csv')
datos_con_D = df[df['annotation'] == 'D']
print(datos_con_D)
datos_con_D.to_csv('datos_con_D.csv', index=False)