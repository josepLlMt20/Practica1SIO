import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #per poder mostrar els grafics
from matplotlib.ticker import FuncFormatter
from collections import defaultdict
import ast

from main import dataBcn

# Configurar pandas para mostrar todas las columnas
pd.set_option('display.max_columns', None)

#Carregar les dades
dataBuAr = pd.read_csv("CityFiles/buenos aires/listings.csv")


columns_to_drop = [
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
    "review_scores_value",
    "price",
    "host_response_rate",
    "host_acceptance_rate",
    "host_identity_verified",
]

columns_non_histogram = [
    "id",
    "host_id",
    "latitude",
    "longitude",
    "name",
    "description",
    "picture_url",
    "neighbourhood",
    "neighbourhood_cleansed",
    "neighbourhood_group_cleansed",
    "property_type",
    "room_type",
    "amenities",
    "instant_bookable"
]

# #Analisis exploratori
# #Eliminar columnas
dataBuArCleaned = dataBuAr.drop(columns= columns_to_drop, axis=1, inplace=False)

# Transformar dades
dataBuArCleaned['price'] = dataBuArCleaned['price'].str.replace('$', '', regex=False)
dataBuArCleaned['price'] = dataBuArCleaned['price'].str.replace(',', '', regex=False)
dataBuArCleaned['price'] = dataBuArCleaned['price'].astype(float)
mean_price = dataBuArCleaned['price'].mean()
dataBuArCleaned['price'].fillna(mean_price, inplace=True)
dataBuArCleaned['host_response_rate'] = dataBuArCleaned['host_response_rate'].str.replace('%', '', regex=False)
dataBuArCleaned['host_response_rate'] = dataBuArCleaned['host_response_rate'].astype(float)
dataBuArCleaned['host_acceptance_rate'] = dataBuArCleaned['host_acceptance_rate'].str.replace('%', '', regex=False)
dataBuArCleaned['host_acceptance_rate'] = dataBuArCleaned['host_acceptance_rate'].astype(float)
dataBuArCleaned['host_identity_verified'] = dataBuArCleaned['host_identity_verified'].replace({'t': True, 'f': False})
dataBuArCleaned['instant_bookable'] = dataBuArCleaned['instant_bookable'].replace({'t': True, 'f': False})

#Gestionem columnes property_type, room_type i amenities. Farem recompte de les diferents opcions que tenim i les agruparem en categories
#Guardarem en un diccionari les diferents opcions que tenim
property_counts = defaultdict(int)
room_counts = defaultdict(int)
amenity_counts = defaultdict(int)

# Funció per comptar elements en una columna
def count_elements(column, counts_dict):
    for value in column:
        if pd.notna(value):  # Ignorar valors NaN
            # Convertir l'string de 'amenities' a llista si es necessari
            if column.name == 'amenities':
                # usem ast.literal_eval per convertir l'string a una lista
                try:
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    value = []
            else:
                value = [value]

            for item in value:
                counts_dict[item] += 1

def format_y_axis(x, pos):
    return f'${int(x):,}'

# Contar en cada columna
count_elements(dataBuArCleaned['property_type'], property_counts)
count_elements(dataBuArCleaned['room_type'], room_counts)
count_elements(dataBuArCleaned['amenities'], amenity_counts)

# Convertir a DataFrames para visualización
property_counts_df = pd.DataFrame(property_counts.items(), columns=['Property Type', 'Count'])
room_counts_df = pd.DataFrame(room_counts.items(), columns=['Room Type', 'Count'])
amenity_counts_df = pd.DataFrame(amenity_counts.items(), columns=['Amenity', 'Count'])

# Mostrar resultados
print("Property Type Counts:")
print(property_counts_df)

print("\nRoom Type Counts:")
print(room_counts_df)

print("\nAmenity Counts:")
print(amenity_counts_df)

# Gràfica Property Types (Top 10)
plt.figure(figsize=(10, 6))
top_property_counts = property_counts_df.sort_values(by='Count', ascending=False).head(10)
plt.barh(top_property_counts['Property Type'], top_property_counts['Count'], color='skyblue')
plt.title('Top 10 Property Types')
plt.xlabel('Count')
plt.ylabel('Property Type')
plt.grid(axis='x')
plt.show()

# Gràfica Room Types
plt.figure(figsize=(10, 6))
plt.barh(room_counts_df['Room Type'], room_counts_df['Count'], color='salmon')
plt.title('Counts of Room Types')
plt.xlabel('Count')
plt.ylabel('Room Type')
plt.grid(axis='x')
plt.show()

# Gràfica Amenities (top 10)
plt.figure(figsize=(10, 6))
top_amenities = amenity_counts_df.sort_values(by='Count', ascending=False).head(10)
plt.barh(top_amenities['Amenity'], top_amenities['Count'], color='lightgreen')
plt.title('Top 10 Amenities Counts')
plt.xlabel('Count')
plt.ylabel('Amenity')
plt.grid(axis='x')
plt.show()


#Instant bookable & number of reviews
grouped_data = dataBuArCleaned.groupby('instant_bookable')['number_of_reviews'].mean().reset_index()

# Gràfica
plt.figure(figsize=(8, 5))
plt.bar(grouped_data['instant_bookable'].astype(str), grouped_data['number_of_reviews'], color=['lightblue', 'salmon'])
plt.title('Average Number of Reviews by Instant Bookable Status')
plt.xlabel('Instant Bookable')
plt.ylabel('Average Number of Reviews')
plt.xticks(ticks=[0, 1], labels=['No', 'Yes'])
plt.grid(axis='y')
plt.show()

#Instant bookable & availability_365
grouped_data = dataBuArCleaned.groupby('instant_bookable')['availability_365'].mean().reset_index()

# Gràfica
plt.figure(figsize=(8, 5))
plt.bar(grouped_data['instant_bookable'].astype(str), grouped_data['availability_365'], color=['lightblue', 'salmon'])
plt.title('Average availability by Instant Bookable Status')
plt.xlabel('Instant Bookable')
plt.ylabel('Average Number of Reviews')
plt.xticks(ticks=[0, 1], labels=['No', 'Yes'])
plt.grid(axis='y')
plt.show()

#Analisi de les dades numeriques

numericDataBuAr = dataBuArCleaned.drop(columns= columns_non_histogram, axis=1, inplace=False)

print("-------------Descripció de les dades:----------------")
print(numericDataBuAr.describe())

#Mediana
median_values = numericDataBuAr.median()
print("-------------Mediana:----------------")
print(median_values)

#Districució de les dades
numericDataBuAr.hist(bins=50, figsize=(20, 15), edgecolor = 'black')
plt.tight_layout()
plt.show()

#Desviació estàndard
std_values = numericDataBuAr.std()
print("-------------Desviació estàndard:----------------")
print(std_values)

#Correlació
correlation_matrix = numericDataBuAr.corr()

#Gràfica
plt.figure(figsize=(12, 8))
plt.matshow(correlation_matrix, cmap='coolwarm', fignum=1)
plt.colorbar(label='Correlation Coefficient')
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation='vertical')
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title('Correlation Matrix')
plt.show()

#Estudi de relacio entre amenities i principals KPIs
#afegim una columna amb el nombre d'amenities
print("-------------Relacions num. amenities i KPIs:----------------")
dataBuArCleaned['amenities_count'] = dataBuArCleaned['amenities'].str.count(',') + 1
# print(dataBuArCleaned[['amenities', 'amenities_count']].head())


# Precio vs. Cantidad de amenidades
plt.figure(figsize=(10, 5))
plt.scatter(dataBuArCleaned['amenities_count'], dataBuArCleaned['price'], color='skyblue')
plt.title('Number of Amenities vs Price')
plt.xlabel('Number of Amenities')
plt.ylabel('Price ($)')
plt.grid(True)
plt.show()

# Reviews vs. Cantidad de amenidades

plt.figure(figsize=(10, 5))
plt.scatter(dataBuArCleaned['amenities_count'], dataBuArCleaned['number_of_reviews'], color='salmon')
plt.title('Number of Amenities vs Number of Reviews')
plt.xlabel('Number of Amenities')
plt.ylabel('Number of Reviews')
plt.grid(True)
plt.show()

#Puntuació vs. Cantidad de amenidades

plt.figure(figsize=(10, 5))
plt.scatter(dataBuArCleaned['amenities_count'], dataBuArCleaned['review_scores_rating'], color='red')
plt.title('Number of Amenities vs Number of Reviews')
plt.xlabel('Number of Amenities')
plt.ylabel('Number of Reviews')
plt.grid(True)
plt.show()


#Estudi de preu, tipo de propietat i puntuació per barris


# Paso 1: Agrupar los datos por barrio y calcular las métricas de interés

# Tipos de alojamiento por barrio (contamos las ocurrencias de cada tipo de alojamiento por barrio)
property_types_by_neighbourhood = dataBuArCleaned.groupby(['neighbourhood_cleansed', 'property_type']).size().unstack(fill_value=0)

# Precio promedio por barrio
price_by_neighbourhood = dataBuArCleaned.groupby('neighbourhood_cleansed')['price'].mean().reset_index()

# Media de reviews por barrio
reviews_by_neighbourhood = dataBuArCleaned.groupby('neighbourhood_cleansed')['review_scores_rating'].mean().reset_index()

# Paso 2: Crear gráficos separados para cada métrica

# Gráfico 1: Tipos de alojamiento por barrio
property_types_by_neighbourhood.plot(kind='bar', stacked=True, figsize=(14, 12), alpha=0.7)
plt.title('Property Types by Neighbourhood')
plt.xlabel('Neighbourhood')
plt.ylabel('Count of Property Types')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Property Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()  # Ajuste automático de márgenes
plt.show()

# Gráfico 2: Precio promedio por barrio
plt.figure(figsize=(14, 6))
plt.bar(price_by_neighbourhood['neighbourhood_cleansed'], price_by_neighbourhood['price'], color='lightgreen')
plt.title('Average Price by Neighbourhood')
plt.xlabel('Neighbourhood')
plt.ylabel('Average Price (ARS)')
plt.xticks(rotation=45, ha='right')
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_y_axis))
plt.tight_layout()
plt.show()

# Gráfico 3: Media de reviews por barrio
plt.figure(figsize=(14, 6))
plt.bar(reviews_by_neighbourhood['neighbourhood_cleansed'], reviews_by_neighbourhood['review_scores_rating'], color='lightcoral')
plt.title('Average Review Scores by Neighbourhood')
plt.xlabel('Neighbourhood')
plt.ylabel('Average Review Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#Gràfica de la relació entre el preu i la puntuació de la review

# Paso 3: Filtrar los datos para que ambos tengan valores no nulos
filtered_df = dataBuArCleaned.dropna(subset=['review_scores_rating'])  # Solo eliminamos nulos en reseñas

# Paso 4: Asegurarse de que no hay NaN o infinitos en el conjunto de datos filtrado
filtered_df = filtered_df[np.isfinite(filtered_df['review_scores_rating']) & np.isfinite(filtered_df['price'])]

# Verificar que hay suficientes datos
if filtered_df.shape[0] < 2:
    raise ValueError("No hay suficientes datos para realizar el ajuste.")

# Paso 5: Crear un gráfico de dispersión
plt.figure(figsize=(12, 6))
plt.scatter(filtered_df['review_scores_rating'], filtered_df['price'], alpha=0.5)
plt.title('Relationship between Price and Review Scores')
plt.xlabel('Review Scores')
plt.ylabel('Price (ARS)')
plt.grid()

# Paso 6: Agregar una línea de tendencia
try:
    m, b = np.polyfit(filtered_df['review_scores_rating'], filtered_df['price'], 1)  # Ajuste lineal
    plt.plot(filtered_df['review_scores_rating'], m * filtered_df['review_scores_rating'] + b, color='red')
except np.linalg.LinAlgError as e:
    print("Error al calcular la línea de tendencia:", e)

plt.tight_layout()
plt.show()