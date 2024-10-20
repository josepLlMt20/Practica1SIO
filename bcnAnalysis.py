import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #per poder mostrar els grafics
import seaborn as sns

#Configurar pandas para mostrar todas las columnas
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
    'reviews_per_month', 'name', 'description', 'picture_url'

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
dataBcn['host_identity_verified'] = dataBcn['host_identity_verified'].str.strip().replace({'t': True, 'f': False})
dataBcn['instant_bookable'] = dataBcn['instant_bookable'].str.strip().replace({'t': True, 'f': False})

#Transformar columna amenities en una lista
dataBcn['amenities'] = dataBcn['amenities'].str.replace('{', '', regex=False)
dataBcn['amenities'] = dataBcn['amenities'].str.replace('}', '', regex=False)
dataBcn['amenities'] = dataBcn['amenities'].str.replace('"', '', regex=False)
dataBcn['amenities'] = dataBcn['amenities'].str.split(',').apply(np.array)


#Mostrar info de las columnas
print(dataBcn[['price','host_response_rate','host_acceptance_rate','host_identity_verified','instant_bookable', 'amenities']].info())

#Guardar los datos transformados
dataBcn.to_csv("CityFiles/barcelona/tarnsformado.csv", sep = ';', index=True)

#Mostrar dataset
#print(dataBcn.head())

# Arrays con las listas de variables
variables_numericas = [
    'host_response_rate', 'host_acceptance_rate',
    'host_total_listings_count', 'accommodates',
    'bathrooms', 'bedrooms', 'beds', 'price', 'minimum_nights', 'maximum_nights',
    'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'availability_365',
    'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy',
    'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication',
    'review_scores_location', 'review_scores_value'
]

variables_categoricas = [
    'neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'property_type',
    'room_type', 'amenities'
]

variables_booleanas = [
    'host_identity_verified', 'instant_bookable'
]
'''
#Analisis variables numeriques
#Estadisticas descriptivas
data_numericas = dataBcn[variables_numericas]
estadisticas_numericas = data_numericas.describe()
print(estadisticas_numericas)



#Distribución de las variables numéricas mas interesantes
#Distribución de la variable price
plt.figure(figsize=(10, 6))
sns.histplot(dataBcn['price'], bins=30, kde=True)
plt.title('Distribución de la variable price')
plt.xlabel('price')
plt.ylabel('Frecuencia')
plt.show()

#Distribución de la variable review_scores_rating
plt.figure(figsize=(10, 6))
sns.histplot(dataBcn['review_scores_rating'], bins=30, kde=True)
plt.title('Distribución de la variable review_scores_rating')
plt.xlabel('review_scores_rating')
plt.ylabel('Frecuencia')
plt.show()

#Distribución de la variable availability_365
plt.figure(figsize=(10, 6))
sns.histplot(dataBcn['availability_365'], bins=30, kde=True)
plt.title('Distribución de la variable availability_365')
plt.xlabel('availability_365')
plt.ylabel('Frecuencia')
plt.show()

#Boxplot de la variable availability_365
plt.figure(figsize=(10, 6))
sns.boxplot(x=dataBcn['availability_365'])
plt.title('Boxplot de la variable availability_365')
plt.xlabel('availability_365')
plt.show()

#Distribución de la variable number_of_reviews
plt.figure(figsize=(10, 6))
sns.histplot(dataBcn['number_of_reviews'], bins=30, kde=True)
plt.title('Distribución de la variable number_of_reviews')
plt.xlabel('number_of_reviews')
plt.ylabel('Frecuencia')
plt.show()

##QUITAR OUTLIERS DE LA VARIABLE PRICE
#Filtrar eliminando outliers usando los percentiles
lower_bound = dataBcn['price'].quantile(0.01)  # 1er percentil
upper_bound = dataBcn['price'].quantile(0.99)  # 99º percentil

# Crear un nuevo dataframe sin outliers
data_filtered = dataBcn[(dataBcn['price'] >= lower_bound) & (dataBcn['price'] <= upper_bound)]

# Representación gráfica sin outliers
sns.histplot(data_filtered['price'], kde=True)
plt.title('Distribución de la variable price sin outliers')
plt.xlabel('price')
plt.ylabel('Frecuencia')
plt.show()

print(data_filtered['price'].describe())

#Boxplot de la variable price
plt.figure(figsize=(10, 6))
sns.boxplot(x=dataBcn['price'])
plt.title('Boxplot de la variable price')
plt.xlabel('price')
plt.show()

#boxplot de la variable number_of_reviews
plt.figure(figsize=(10, 6))
sns.boxplot(x=dataBcn['number_of_reviews'])
plt.title('Boxplot de la variable number_of_reviews')
plt.xlabel('number_of_reviews')
plt.show()

#Histograma de la variable number_of_reviews
plt.figure(figsize=(10, 6))
sns.histplot(dataBcn['number_of_reviews'], bins=30, kde=True)
plt.title('Distribución de la variable number_of_reviews')
plt.xlabel('number_of_reviews')
plt.ylabel('Frecuencia')
plt.show()

#QUITAR OUTLIERS DE LA VARIABLE number_of_reviews
#Filtrar eliminando outliers usando los percentiles
lower_bound = dataBcn['number_of_reviews'].quantile(0.25)  # 1er quartil
upper_bound = dataBcn['number_of_reviews'].quantile(0.75)  # 3 quartil

# Crear un nuevo dataframe sin outliers
data_filtered = dataBcn[(dataBcn['number_of_reviews'] >= lower_bound) & (dataBcn['number_of_reviews'] <= upper_bound)]

# Representación gráfica sin outliers
sns.histplot(data_filtered['number_of_reviews'], kde=True)
plt.title('Distribución de la variable number_of_reviews sin outliers')
plt.xlabel('number_of_reviews')
plt.ylabel('Frecuencia')
plt.show()

# Representació histograma de la variable accommodates
plt.figure(figsize=(10, 6))
sns.histplot(dataBcn['accommodates'], bins=15, kde=True)
plt.title('Distribución de la variable accommodates')
plt.xlabel('accommodates')
plt.ylabel('Frecuencia')
plt.show()
'''


#Analisis variables categoricas
#Estadisticas descriptivas
data_categoricas = dataBcn[variables_categoricas]

frequencies = data_categoricas['neighbourhood_group_cleansed'].value_counts()  # Frecuencia absoluta
relative_frequencies = data_categoricas['neighbourhood_group_cleansed'].value_counts(normalize=True)  # Frecuencia relativa
percentage = relative_frequencies * 100  # Porcentaje

# Crear un DataFrame con estos valores
tabla_frecuencias = pd.DataFrame({
    'Frecuencia': frequencies,
    'Frecuencia Relativa': relative_frequencies,
    'Porcentaje (%)': percentage
})

# Mostrar la tabla
print(tabla_frecuencias)

# Representación gráfica de la variable neighbourhood_group_cleansed
plt.figure(figsize=(10, 6))
sns.countplot(x='neighbourhood_group_cleansed', data=data_categoricas)
plt.title('Distribución de la variable neighbourhood_group_cleansed')
plt.xlabel('neighbourhood_group_cleansed')
plt.ylabel('Frecuencia')
plt.xticks(rotation=45)
plt.show()


data_categoricas['neighbourhood_group_cleansed'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Distribución de la variable neighbourhood_group_cleansed')
plt.ylabel('')
plt.show()