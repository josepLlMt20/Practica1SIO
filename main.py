import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # para poder mostrar los gráficos

# Cargar los datos
dataBcn = pd.read_csv("CityFiles/barcelona/listings.csv")
dataBer = pd.read_csv("CityFiles/berlin/listings.csv")
dataBuAr = pd.read_csv("CityFiles/buenos aires/listings.csv")

# Eliminar columnas irrelevantes
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
    'reviews_per_month', 'description'
]


# Limpiar y procesar cada conjunto de datos
def procesar_datos(data):
    # Eliminar columnas irrelevantes
    data.drop(columns=columnas_a_eliminar, axis=1, inplace=True)

    # Eliminar caracteres no numéricos de la columna price y convertir a float
    data['price'] = pd.to_numeric(data['price'].replace(r'[\$,]', '', regex=True), errors='coerce')

    return data


# Procesar los datos de cada ciudad
dataBcn = procesar_datos(dataBcn)
dataBer = procesar_datos(dataBer)
dataBuAr = procesar_datos(dataBuAr)

# Guardar los archivos transformados
dataBcn.to_csv("CityFiles/barcelona/transformado.csv", sep=';', index=False)
dataBuAr.to_csv("CityFiles/buenos aires/transformado.csv", sep=';', index=False)
dataBer.to_csv("CityFiles/berlin/transformado.csv", sep=';', index=False)
