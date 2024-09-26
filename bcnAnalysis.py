import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #per poder mostrar els grafics

# Configurar pandas para mostrar todas las columnas
pd.set_option('display.max_columns', None)

#Carregar les dades
dataBcn = pd.read_csv("CityFiles/barcelona/listings.csv")

# Eliminar las columnas no deseadas
columnas_a_eliminar = [
    'listing_url', 'scrape_id', 'last_scraped', 'source', 'neighborhood_overview',
    'host_url', 'host_name', 'host_since', 'host_location', 'host_about',
    'host_response_time', 'host_is_superhost', 'host_thumbnail_url', 'host_picture_url',
    'host_neighbourhood', 'host_listings_count', 'host_verifications', 'host_has_profile_pic',
    'bathrooms_text', 'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights',
    'maximum_maximum_nights', 'calendar_updated', 'has_availability', 'availability_30',
    'availability_60', 'availability_90', 'calendar_last_scraped', 'number_of_reviews_l30d',
    'number_of_reviews_ltm', 'first_review', 'last_review', 'license',
    'calculated_host_listings_count', 'calculated_host_listings_count_entire_homes',
    'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms',
    'reviews_per_month'
]

#Eliminar columnas que son irrelevantes
dataBcn.drop(columns=columnas_a_eliminar, axis=1, inplace=True)


#Parte transformación datos
#Pasar a float
dataBcn['price'] = dataBcn['price'].str.replace('$', '', regex=False)
dataBcn['price'] = dataBcn['price'].str.replace(',', '', regex=False)
dataBcn['price'] = dataBcn['price'].astype(float)
dataBcn['host_response_rate'] = dataBcn['host_response_rate'].str.replace('%', '', regex=False)
dataBcn['host_response_rate'] = dataBcn['host_response_rate'].astype(float)
dataBcn['host_acceptance_rate'] = dataBcn['host_acceptance_rate'].str.replace('%', '', regex=False)
dataBcn['host_acceptance_rate'] = dataBcn['host_acceptance_rate'].astype(float)

#Pasar a boleano
dataBcn['host_identity_verified'] = dataBcn['host_identity_verified'].replace({'t': True, 'f': False})
dataBcn['instant_bookable'] = dataBcn['instant_bookable'].replace({'t': True, 'f': False})

# Verifica el resultado
print(dataBcn[['price','host_response_rate','host_acceptance_rate','host_identity_verified','instant_bookable']].info())





#Columnas que se usarn para el analisis
#print("Columnes per fer l'analisis:")
#print(dataBcn.columns)

#dataBcn.hist('availability_365')
#plt.show()


#Histograma relaciones
#
#dataBcn.plot(kind='scatter', x='accommodates', y='availability_365', alpha=0.5)
#plt.title("Relación entre el Precio y la Disponibilidad")
#plt.xlabel("Accommodates")
#plt.ylabel("Días disponibles al año")
#plt.show()





#Per passar-ho a csv
#dataBcn.to_csv("CityFiles/barcelona/tarnsformado.csv", sep = ';', index=True)




