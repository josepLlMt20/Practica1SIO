import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #per poder mostrar els grafics

# Configurar pandas para mostrar todas las columnas
#pd.set_option('display.max_columns', None)

#Carregar les dades
dataBcn = pd.read_csv("CityFiles/barcelona/listings.csv")
dataBer = pd.read_csv("CityFiles/berlin/listings.csv")
dataBuAr = pd.read_csv("CityFiles/buenos_aires/listings.csv")

print(dataBcn.head())
print(dataBcn.info())

dataBcn.hist('availability_365')
plt.show()

#
#
# #Histograma relaciones
# dataBcn.plot(kind='scatter', x='accommodates', y='availability_365', alpha=0.5)
# plt.title("Relación entre el Precio y la Disponibilidad")
# plt.xlabel("Accommodates")
# plt.ylabel("Días disponibles al año")
# plt.show()

dataBcn.to_csv("CityFiles/barcelona/tarnsformado.csv", sep = ';', index=True)