import ast
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar pandas para mostrar todas las columnas
pd.set_option('display.max_columns', None)

# Cargar los datos
dataBcn = pd.read_csv("CityFiles/barcelona/listings.csv")

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
    'reviews_per_month', 'name', 'description', 'picture_url'
]
dataBcn.drop(columns=columnas_a_eliminar, axis=1, inplace=True)

# Transformaciones de datos
# Convertir precios y tasas a formato float
dataBcn['price'] = dataBcn['price'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)
dataBcn['host_response_rate'] = dataBcn['host_response_rate'].str.replace('%', '', regex=False).astype(float)
dataBcn['host_acceptance_rate'] = dataBcn['host_acceptance_rate'].str.replace('%', '', regex=False).astype(float)

# Convertir a booleano
dataBcn['host_identity_verified'] = dataBcn['host_identity_verified'].str.strip().replace({'t': True, 'f': False})
dataBcn['instant_bookable'] = dataBcn['instant_bookable'].str.strip().replace({'t': True, 'f': False})

# Transformar columna amenities en una lista
dataBcn['amenities'] = dataBcn['amenities'].str.replace('{', '', regex=False).str.replace('}', '', regex=False).str.replace('"', '', regex=False).str.split(',').apply(np.array)

# Mostrar información de las columnas
print(dataBcn[['price', 'host_response_rate', 'host_acceptance_rate', 'host_identity_verified', 'instant_bookable', 'amenities']].info())

# Guardar los datos transformados
dataBcn.to_csv("CityFiles/barcelona/transformado.csv", sep=';', index=True)

# Mostrar el dataset
print(dataBcn.head())

# Variables para análisis
variables_numericas = [
    'host_response_rate', 'host_acceptance_rate', 'host_total_listings_count', 'accommodates',
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

# Análisis de variables numéricas
# Estadísticas descriptivas
data_numericas = dataBcn[variables_numericas]
estadisticas_numericas = data_numericas.describe()
print(estadisticas_numericas)

# Distribución de la variable 'price'
plt.figure(figsize=(10, 6))
sns.histplot(dataBcn['price'], bins=30, kde=True)
plt.title('Distribución de la variable price')
plt.xlabel('Price')
plt.ylabel('Frecuencia')
plt.show()

# Distribución de 'review_scores_rating'
plt.figure(figsize=(10, 6))
sns.histplot(dataBcn['review_scores_rating'], bins=30, kde=True)
plt.title('Distribución de la variable review_scores_rating')
plt.xlabel('Rating')
plt.ylabel('Frecuencia')
plt.show()

# Distribución de 'availability_365'
plt.figure(figsize=(10, 6))
sns.histplot(dataBcn['availability_365'], bins=30, kde=True)
plt.title('Distribución de la variable availability_365')
plt.xlabel('Availability 365')
plt.ylabel('Frecuencia')
plt.show()

# Boxplot de 'availability_365'
plt.figure(figsize=(10, 6))
sns.boxplot(x=dataBcn['availability_365'])
plt.title('Boxplot de la variable availability_365')
plt.xlabel('Availability 365')
plt.show()

# Eliminar outliers en 'price'
lower_bound_price = dataBcn['price'].quantile(0.01)
upper_bound_price = dataBcn['price'].quantile(0.99)
data_filtered_price = dataBcn[(dataBcn['price'] >= lower_bound_price) & (dataBcn['price'] <= upper_bound_price)]

# Representación gráfica sin outliers de 'price'
sns.histplot(data_filtered_price['price'], kde=True)
plt.title('Distribución de la variable price sin outliers')
plt.xlabel('Price')
plt.ylabel('Frecuencia')
plt.show()

# Análisis de variables categóricas
data_categoricas = dataBcn[variables_categoricas]

# Frecuencias y porcentaje de 'neighbourhood_group_cleansed'
frequencies = data_categoricas['neighbourhood_group_cleansed'].value_counts()
relative_frequencies = data_categoricas['neighbourhood_group_cleansed'].value_counts(normalize=True) * 100
tabla_frecuencias = pd.DataFrame({
    'Frecuencia': frequencies,
    'Porcentaje (%)': relative_frequencies
})
print(tabla_frecuencias)

# Representación gráfica de 'neighbourhood_group_cleansed'
plt.figure(figsize=(10, 6))
sns.countplot(x='neighbourhood_group_cleansed', data=data_categoricas)
plt.title('Distribución de la variable neighbourhood_group_cleansed')
plt.xlabel('Neighbourhood Group')
plt.ylabel('Frecuencia')
plt.xticks(rotation=45)
plt.show()

# Pie chart de 'neighbourhood_group_cleansed'
data_categoricas['neighbourhood_group_cleansed'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Distribución de la variable neighbourhood_group_cleansed')
plt.ylabel('')
plt.show()

# Neighbourhoods vs Price
# Generar el boxplot con la variable price sin outliers
plt.figure(figsize=(15, 12))
sns.boxplot(x='neighbourhood_group_cleansed', y='price', data=data_filtered_price)
plt.title('Comparación de Price sin outliers vs Neighbourhood Group Cleansed')
plt.xlabel('Neighbourhood Group Cleansed')
plt.ylabel('Price')
plt.xticks(rotation=45)  # Rotar las etiquetas del eje X si es necesario
plt.show()

# Availability vs Price
# Generar el scatter plot de availability_365 vs price
plt.figure(figsize=(10, 6))
sns.scatterplot(x='price', y='availability_365', data=data_filtered_price)
plt.title('Scatter plot de Availability 365 vs Price')
plt.xlabel('Price')
plt.ylabel('Availability 365')
plt.show()
correlation = data_filtered_price['price'].corr(data_filtered_price['availability_365'])
print(f'La correlación de Pearson entre Price y availability es: {correlation}')

# Price vs Review Scores Rating
# Crear el scatter plot de Price vs review_scores_rating
plt.figure(figsize=(10, 6))
sns.scatterplot(x='price', y='review_scores_rating', data=data_filtered_price)
plt.title('Scatter plot de Price vs Review Scores Rating')
plt.xlabel('Price')
plt.ylabel('Review Scores Rating')
plt.show()

correlation = data_filtered_price['price'].corr(data_filtered_price['review_scores_rating'])
print(f'La correlación de Pearson entre Price y Review Scores Rating es: {correlation}')

# Property Type vs Neighbourhood Group Cleansed
# Crear una tabla de frecuencia de Property Type vs Neighbourhood Group Cleansed
property_neighbourhood_table = pd.crosstab(dataBcn['property_type'], dataBcn['neighbourhood_group_cleansed'])

# Crear el heatmap
plt.figure(figsize=(20, 15))
sns.heatmap(property_neighbourhood_table, cmap='Blues', annot=False, linewidths=0.5)

# Personalizar el gráfico
plt.title('Heatmap de Property Type vs Neighbourhood Group Cleansed')
plt.xlabel('Neighbourhood Group Cleansed')
plt.ylabel('Property Type')
plt.xticks(rotation=90)  # Rotar etiquetas del eje X si es necesario
plt.show()

# Crear el gráfico boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='instant_bookable', y='availability_365', data=dataBcn)
plt.title('Instant Bookable vs Availability 365')
plt.xlabel('Instant Bookable')
plt.ylabel('Availability 365 (días)')
plt.show()

# instant_bookable vs price
plt.figure(figsize=(10, 6))
sns.boxplot(x='instant_bookable', y='price', data=data_filtered_price)
plt.title('Instant Bookable vs price')
plt.xlabel('Instant Bookable')
plt.ylabel('Price')
plt.show()

# Avaibility vs property_type
top10_PropertyType = dataBcn['property_type'].value_counts().head(10).index
top10PT = dataBcn[dataBcn['property_type'].isin(top10_PropertyType)]

plt.figure(figsize=(15, 10))
sns.barplot(x='availability_365', y='property_type', data=top10PT)
plt.title('Property Type vs Availability 365')
plt.xlabel('Availability 365')
plt.ylabel('Property Type')
plt.show()

#Per contar amenities
amenity_counts = defaultdict(int)

#Funció per netejar i comptar elements
def count_elements(column, counts_dict):
    for value in column.dropna():
        try:
            #Si el valor es un numpy array, convertirlo a string
            if isinstance(value, np.ndarray):
                value = ' '.join(value)

            #Limpiar y normalizar el string
            value = value.replace('\\u2019', "'")  #Reemplazar caracteres unicode
            value = value.strip('[]')  # Eliminar corchetes
            value = value.replace('\'', '').replace('\"', '')
            value = value.split('  ')  # Dividir por dobles espacios o comas

            # Eliminar espacios en blanco adicionales
            value = [item.strip() for item in value if item.strip()]
        except (ValueError, SyntaxError):  # Si hay error, tratamos como lista vacía
            value = []

        #Incrementar el contador por cada item en la lista
        for item in value:
            counts_dict[item] += 1


# Contar los elementos en la columna 'amenities'
count_elements(dataBcn['amenities'], amenity_counts)

# Convertir a DataFrame para visualización
amenity_counts_df = pd.DataFrame(amenity_counts.items(), columns=['Amenity', 'Count'])

# Ordenar por contador
amenity_counts_df = amenity_counts_df.sort_values(by='Count', ascending=False)

# Gráfica de las 10 principales amenities
plt.figure(figsize=(15, 10))
top_amenities = amenity_counts_df.head(10)
plt.barh(top_amenities['Amenity'], top_amenities['Count'])
plt.title('Top 10 Amenities Counts')
plt.xlabel('Count')
plt.ylabel('Amenity')
plt.grid(axis='x')
plt.show()

# Mostrar el DataFrame con el conteo de amenities
print(amenity_counts_df)

# Añadir una nueva columna con la cantidad de amenities
dataBcn['amenities_count'] = dataBcn['amenities'].apply(lambda x: len(x))

# Filtrar solo las columnas que necesitas: amenities_count y price
data_filtered_price = dataBcn[['amenities_count', 'price']]

# Crear el scatter plot de amenities_count vs price
plt.figure(figsize=(10, 6))
sns.scatterplot(x='amenities_count', y='price', data=data_filtered_price)
plt.title('Scatter plot de Amenities Count vs Price')
plt.xlabel('Amenities Count')
plt.ylabel('Price')
plt.show()

# Calcular la correlación de Pearson entre amenities_count y price
correlation = data_filtered_price['amenities_count'].corr(data_filtered_price['price'])
print(f'La correlación de Pearson entre Amenities Count y Price es: {correlation}')

#Cantidad amenities vs property_type
top10_PropertyType = dataBcn['property_type'].value_counts().head(10).index
top10PT = dataBcn[dataBcn['property_type'].isin(top10_PropertyType)]

plt.figure(figsize=(15, 15))
sns.boxplot(x='property_type', y='amenities_count', data=top10PT)
plt.title('Property Type vs Amenities Count')
plt.xlabel('Property Type')
plt.ylabel('Amenities Count')
plt.xticks(rotation=45)
plt.show()

