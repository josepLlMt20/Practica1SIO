import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #per poder mostrar els grafics

# Configurar pandas para mostrar todas las columnas
#pd.set_option('display.max_columns', None)

#Carregar les dades
dataBuAr = pd.read_csv("CityFiles/buenos aires/listings.csv")


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

columns_histogram = [
    "host_total_listings_count",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "minimum_nights",
    "maximum_nights",
    "minimum_nights_avg_ntm",
    "maximum_nights_avg_ntm",
    "availability_365",
    "number_of_reviews",
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value"
]

columns_non_histogram = [
    "name",
    "description",
    "picture_url",
    "host_response_rate",
    "host_acceptance_rate",
    "host_identity_verified",
    "neighbourhood",
    "neighbourhood_cleansed",
    "neighbourhood_group_cleansed",
    "property_type",
    "room_type",
    "amenities",
    "price",
    "instant_bookable"
]


#Analisis exploratori
#Eliminar columnas
dataBuArCleaned = dataBuAr.drop(columns= columnas_a_eliminar, axis=1, inplace=False)
print(dataBuArCleaned.info())

#Parte transformación datos
#Pasar a float
dataBuArCleaned['price'] = dataBuArCleaned['price'].str.replace('$', '', regex=False)
dataBuArCleaned['price'] = dataBuArCleaned['price'].str.replace(',', '', regex=False)
dataBuArCleaned['price'] = dataBuArCleaned['price'].astype(float)
dataBuArCleaned['host_response_rate'] = dataBuArCleaned['host_response_rate'].str.replace('%', '', regex=False)
dataBuArCleaned['host_response_rate'] = dataBuArCleaned['host_response_rate'].astype(float)
dataBuArCleaned['host_acceptance_rate'] = dataBuArCleaned['host_acceptance_rate'].str.replace('%', '', regex=False)
dataBuArCleaned['host_acceptance_rate'] = dataBuArCleaned['host_acceptance_rate'].astype(float)

#Pasar a boleano
dataBuArCleaned['host_identity_verified'] = dataBuArCleaned['host_identity_verified'].replace({'t': True, 'f': False})
dataBuArCleaned['instant_bookable'] = dataBuArCleaned['instant_bookable'].replace({'t': True, 'f': False})

# # Verifica el resultado
# print(dataBuArCleaned[['price','host_response_rate','host_acceptance_rate','host_identity_verified','instant_bookable']].info())

for column in columns_histogram:
    dataBuArCleaned.hist(column)
    plt.show()

for column in columns_histogram:
    print(f"Estadísticas para la columna {column}:")
    print(dataBuArCleaned[column].describe())
# dataBuAr.to_csv("CityFiles/barcelona/tarnsformado.csv", sep = ';', index=True)