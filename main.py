import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #per poder mostrar els grafics

#Carregar les dades
dataBcn = pd.read_csv("CityFiles/barcelona/listings.csv")
dataBer = pd.read_csv("CityFiles/berlin/listings.csv")
dataBuAr = pd.read_csv("CityFiles/buenos aires/listings.csv")

print(dataBcn.columns)
print(dataBcn.dtypes)

dataBcn.hist('availability_365')
plt.show()
