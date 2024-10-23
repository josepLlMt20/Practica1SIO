import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #per poder mostrar els grafics
from matplotlib.ticker import FuncFormatter
from collections import defaultdict
import ast
import seaborn as sns

# Configurar pandas para mostrar todas las columnas
pd.set_option('display.max_columns', None)

#Carregar les dades
dataBer = pd.read_csv("CityFiles/berlin/listings.csv")

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

# #Analisis exploratori
# #Eliminar columnas
dataBerCleaned = dataBer.drop(columns= columns_to_drop, axis=1, inplace=False)

# Transformar dades
dataBerCleaned['price'] = dataBerCleaned['price'].str.replace('$', '', regex=False)
dataBerCleaned['price'] = dataBerCleaned['price'].str.replace(',', '', regex=False)
dataBerCleaned['price'] = dataBerCleaned['price'].astype(float)
mean_price = dataBerCleaned['price'].mean()
dataBerCleaned['price'].fillna(mean_price, inplace=True)
mean_review_scores_rating = dataBerCleaned['review_scores_rating'].mean()
dataBerCleaned['review_scores_rating'].fillna(mean_review_scores_rating, inplace=True)

#Gestionem columnes property_type, room_type i amenities. Farem recompte de les diferents opcions que tenim i les agruparem en categories
#Guardarem en un diccionari les diferents opcions que tenim
amenity_counts = defaultdict(int)

print('Total de dades: ' + str(len(dataBerCleaned)))

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


# Contar amenities
count_elements(dataBerCleaned['amenities'], amenity_counts)
amenity_counts_df = pd.DataFrame(amenity_counts.items(), columns=['Amenity', 'Count'])

print("\nAmenity Counts:")
print(amenity_counts_df)

# Gràfica Amenities (top 10)
plt.figure(figsize=(10, 6))
top_amenities = amenity_counts_df.sort_values(by='Count', ascending=False).head(10)
plt.barh(top_amenities['Amenity'], top_amenities['Count'], color='lightgreen')
plt.title('Top 10 Amenities')
plt.xlabel('Freq.')
plt.ylabel('Amenity')
plt.grid(axis='x')
plt.show()



# Contar cuántos listados hay por cada grupo de vecindarios
neighbourhood_group_counts = dataBerCleaned['neighbourhood_group_cleansed'].value_counts()

top_2_neighbourhoods = neighbourhood_group_counts.head(2) # Obtener los dos barrios más frecuentes
total_listings = neighbourhood_group_counts.sum()

# Calcular el porcentaje que representan los dos barrios más frecuentes
percentage_top_2 = (top_2_neighbourhoods.sum() / total_listings) * 100
print(f'Los dos barrios más frecuentes representan el {percentage_top_2:.2f}% del total de listados.')

# Ajustar el tamaño del gráfico
plt.figure(figsize=(12, 6))

# Crear gráfico de barras
plt.bar(neighbourhood_group_counts.index, neighbourhood_group_counts.values, color='skyblue')

# Añadir título y etiquetas
plt.title('Distribució segons el grup barri', fontsize=16)
plt.xlabel('Grup barri', fontsize=14)
plt.ylabel('Freq.', fontsize=14)

# Rotar etiquetas del eje x para mejorar la legibilidad
plt.xticks(rotation=45, ha='right', fontsize=12)

# Mostrar gráfico
plt.tight_layout()  # Ajustar el layout para evitar solapamiento
plt.show()



# Contar cuántos listados hay por cada tipo de habitacion
room_type_counts = dataBerCleaned['room_type'].value_counts()

top2_room_type_counts = room_type_counts.head(2) # Obtener los dos barrios más frecuentes
total_listings = room_type_counts.sum()

# Calcular el porcentaje que representan los dos barrios más frecuentes
percentage_top_2 = (top2_room_type_counts.sum() / total_listings) * 100
print(f'Los dos tipos de hab. más frecuentes representan el {percentage_top_2:.2f}% del total de listados.')

# Ajustar el tamaño del gráfico
plt.figure(figsize=(12, 6))

# Crear gráfico de barras
plt.bar(room_type_counts.index, room_type_counts.values, color='skyblue')

# Añadir título y etiquetas
plt.title("Distribució segons el tipus d'habitació", fontsize=16)
plt.xlabel("Tipus d'habitació", fontsize=14)
plt.ylabel('Freq.', fontsize=14)

# Rotar etiquetas del eje x para mejorar la legibilidad
plt.xticks(rotation=45, ha='right', fontsize=12)

# Mostrar gráfico
plt.tight_layout()  # Ajustar el layout para evitar solapamiento
plt.show()



# Contar cuántos listados hay por cada tipo de propiedad y mostrar top 10
property_type_counts = dataBerCleaned['property_type'].value_counts().head(10)

# Ajustar el tamaño del gráfico
plt.figure(figsize=(12, 6))

# Crear gráfico de barras
plt.bar(property_type_counts.index, property_type_counts.values, color='skyblue')

# Añadir título y etiquetas
plt.title('Distribució segons el tipus de propietat', fontsize=16)
plt.xlabel('Tipus de propietat', fontsize=14)
plt.ylabel('Freq.', fontsize=14)

# Rotar etiquetas del eje x para mejorar la legibilidad
plt.xticks(rotation=45, ha='right', fontsize=12)

# Mostrar gráfico
plt.tight_layout()  # Ajustar el layout para evitar solapamiento
plt.show()

# Cálculo del porcentaje
total_properties = property_type_counts.sum()
top_5_properties = property_type_counts.head(5)
others = total_properties - top_5_properties.sum()

# Crear un nuevo DataFrame con "Altres" usando pd.concat
property_type_with_others = pd.concat([top_5_properties, pd.Series({'Altres': others})])

# Gráfico de porcentaje
plt.figure(figsize=(8, 8))
plt.pie(property_type_with_others, labels=property_type_with_others.index, autopct='%1.1f%%', startangle=140, colors=['skyblue', 'lightgreen', 'lightcoral', 'lightpink', 'orange', 'lightgrey'])
plt.title('Distribució percentual segons el tipus de propietat (Top 5 + Altres)')
plt.axis('equal')
plt.tight_layout()
plt.show()



# Contar cuántos listados hay por cada propietario y mostrar top 10
total_hosts = dataBerCleaned['host_id'].value_counts()
host_id_counts = total_hosts.head(10)

# Obtener el top 1 de hosts
top_host = total_hosts.head(1)
total_listings = total_hosts.sum()

# Calcular la frecuencia y el porcentaje del top 1
top_host_frequency = top_host.values[0]
top_host_percentage = (top_host_frequency / total_listings) * 100

# Mostrar resultados
print(f'Frecuencia del top 1 host: {top_host_frequency}')
print(f'Porcentaje del top 1 host: {top_host_percentage:.2f}%')

# Ajustar el tamaño del gráfico
plt.figure(figsize=(12, 6))

# Convertir los índices de host_id a string para el gráfico
host_ids_str = host_id_counts.index.astype(str)

# Crear gráfico de barras
plt.bar(host_ids_str, host_id_counts.values, color='skyblue')

# Añadir título y etiquetas
plt.title('Distribució segons el propietari', fontsize=16)
plt.xlabel('Propietario (ID)', fontsize=14)
plt.ylabel('Freq.', fontsize=14)

# Rotar etiquetas del eje x para mejorar la legibilidad
plt.xticks(rotation=45, ha='right', fontsize=12)

# Mostrar gráfico
plt.tight_layout()  # Ajustar el layout para evitar solapamiento
plt.show()

# Cálculo del porcentaje
hosts_dict = dataBerCleaned['host_id'].value_counts()
total_hosts = hosts_dict.sum()

# Función para crear gráficos de pastel
def plot_pie_chart(percent, hosts_dict, total_hosts):
    # Calcular el número de propietarios que representan el porcentaje especificado
    percent_hosts_count = int(len(hosts_dict) * percent)

    # Obtener los propietarios más frecuentes (top percent)
    percent_hosts_dict = hosts_dict.nlargest(percent_hosts_count)

    # Calcular la cantidad total de listados de los propietarios en el top percent
    top_percent_total = percent_hosts_dict.sum()

    # Calcular "Altres" (total - total de top percent)
    others_count = total_hosts - top_percent_total

    # Crear un nuevo DataFrame con "Altres"
    host_counts_with_others = pd.Series(
        {'Top {}% Propietaris'.format(int(percent * 100)): top_percent_total, 'Altres': others_count})

    # Gráfico de porcentaje
    plt.figure(figsize=(6, 6))  # Tamaño reducido del gráfico
    plt.pie(host_counts_with_others, labels=host_counts_with_others.index, autopct='%1.1f%%', startangle=140,
            colors=['skyblue', 'lightgreen'], textprops={'fontsize': 16})  # Texto más grande
    plt.title('Distribució percentual segons el propietari (Top {}% + Altres)'.format(int(percent * 100)), fontsize=18)
    plt.axis('equal')  # Para que el gráfico sea un círculo
    plt.tight_layout()

    # Mostrar total de propiedades
    plt.figtext(0.5, 0.01, f'Total de Propiedades: {total_hosts}', ha='center', fontsize=14)  # Texto más grande
    plt.show()

# Graficar para 1%, 5%, y 10%
plot_pie_chart(0.01, hosts_dict, total_hosts)
plot_pie_chart(0.05, hosts_dict, total_hosts)
plot_pie_chart(0.10, hosts_dict, total_hosts)




# Seleccionar les columnes d'interès
variables_of_interest = ['price', 'availability_365', 'number_of_reviews', 'accommodates']
dataBerExploratory = dataBerCleaned[variables_of_interest]

# Calcular les estadístiques descriptives
descriptive_stats = dataBerExploratory.describe(percentiles=[0.25, 0.5, 0.75])
print("Estadístiques descriptives:")
print(descriptive_stats)

# Crear histogramas per cada variable abans de tractar els outliers
plt.figure(figsize=(15, 10))
for i, column in enumerate(variables_of_interest):
    plt.subplot(2, 2, i + 1)
    sns.histplot(dataBerExploratory[column], bins=30, kde=True)
    plt.title(f'Distribució de {column} (Abans de tractar outliers)')
    plt.xlabel(column)
    plt.ylabel('Freqüència')
plt.tight_layout()
plt.show()

# Crear boxplots per cada variable abans de tractar els outliers
plt.figure(figsize=(15, 10))
for i, column in enumerate(variables_of_interest):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(data=dataBerExploratory, y=column)
    plt.title(f'Boxplot de {column} (Abans de tractar outliers)')
plt.tight_layout()
plt.show()

# Funció per identificar outliers basat en l'IQR
def identify_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 2 * IQR
    upper_bound = Q3 + 2 * IQR
    return (df[column] < lower_bound) | (df[column] > upper_bound)

# Identificar outliers i substituir-los per la mitjana
dataBerNoOutliers = dataBerCleaned.copy()

for column in variables_of_interest:
    outliers_mask = identify_outliers_iqr(dataBerNoOutliers, column)
    mean_value = dataBerNoOutliers[column].mean()
    dataBerNoOutliers.loc[outliers_mask, column] = mean_value
    outliers_count = outliers_mask.sum()
    total_rows = dataBerExploratory.shape[0]
    print(f'Outliers substituïts per {column}: {outliers_count} ({(outliers_count / total_rows) * 100:.2f}%)')

# Crear histogramas per cada variable després de substituir els outliers per la mitjana
plt.figure(figsize=(15, 10))
for i, column in enumerate(variables_of_interest):
    plt.subplot(2, 2, i + 1)
    sns.histplot(dataBerNoOutliers[column], bins=30, kde=True)
    plt.title(f'Distribució de {column} (Després de substituir outliers per la mitjana)')
    plt.xlabel(column)
    plt.ylabel('Freqüència')
plt.tight_layout()
plt.show()

# Crear boxplots per cada variable després de substituir els outliers per la mitjana
plt.figure(figsize=(15, 10))
for i, column in enumerate(variables_of_interest):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(data=dataBerNoOutliers, y=column)
    plt.title(f'Boxplot de {column} (Després de substituir outliers per la mitjana)')
plt.tight_layout()
plt.show()

# Mostrar el conjunt de dades després de tractar els outliers
print("\nDades després de tractar els outliers:")
print(dataBerNoOutliers.describe())


# Seleccionar només les columnes de puntuació
scores_columns = [
    'review_scores_rating',
    'review_scores_accuracy',
    'review_scores_cleanliness',
    'review_scores_checkin',
    'review_scores_communication',
    'review_scores_location',
    'review_scores_value'
]

# Renombrar les columnes per a etiquetes més curtes
short_labels = {
    'review_scores_rating': 'Rating',
    'review_scores_accuracy': 'Accuracy',
    'review_scores_cleanliness': 'Cleanliness',
    'review_scores_checkin': 'Check-in',
    'review_scores_communication': 'Communication',
    'review_scores_location': 'Location',
    'review_scores_value': 'Value'
}

# Crear un nou DataFrame només amb aquestes columnes
scores_data = dataBerCleaned[scores_columns]

# Renombrar les columnes al DataFrame
scores_data.rename(columns=short_labels, inplace=True)

# Tornar a calcular la matriu de correlació amb els nous noms
correlation_matrix = scores_data.corr()

plt.figure(figsize=(10, 6))  # Augmenta la mida de la figura si cal
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

plt.title('Correlació entre les puntuacions de ressenyes')
plt.xticks(rotation=45, ha='right')  # Rotar les etiquetes per evitar que es tallin
plt.yticks(rotation=0)

# Ajustar automàticament el layout per evitar que les etiquetes es tallin
plt.tight_layout()

plt.show()

"""

A PARTIR D'AQUIIIIII ASLDLADSFJADSFKDHKFLJASHFKSDKASKLASDFSHKSDF

"""

# Calcular el precio promedio por barrio y ordenarlo
mean_price_by_neighbourhood = dataBerNoOutliers.groupby('neighbourhood_group_cleansed')['price'].mean().reset_index()
mean_price_by_neighbourhood = mean_price_by_neighbourhood.sort_values(by='price', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=mean_price_by_neighbourhood, x='neighbourhood_group_cleansed', y='price', ci=None)
plt.title('Precio promedio por Barrio')
plt.xlabel('Barrio')
plt.ylabel('Precio promedio')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Calcular el precio promedio por tipo de habitación y ordenarlo
mean_price_by_room_type = dataBerNoOutliers.groupby('room_type')['price'].mean().reset_index()
mean_price_by_room_type = mean_price_by_room_type.sort_values(by='price', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=mean_price_by_room_type, x='room_type', y='price', ci=None)
plt.title('Precio promedio por Tipo de Habitación')
plt.xlabel('Tipo de Habitación')
plt.ylabel('Precio promedio')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Calcular el precio promedio por tipo de propiedad (Top 10) y ordenarlo
top_property_types = dataBerNoOutliers['property_type'].value_counts().head(10).index
filtered_data_property_type = dataBerNoOutliers[dataBerNoOutliers['property_type'].isin(top_property_types)]
mean_price_by_property_type = filtered_data_property_type.groupby('property_type')['price'].mean().reset_index()
mean_price_by_property_type = mean_price_by_property_type.sort_values(by='price', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=mean_price_by_property_type, x='property_type', y='price', ci=None)
plt.title('Precio promedio por Tipo de Propiedad (Top 10)')
plt.xlabel('Tipo de Propiedad')
plt.ylabel('Precio promedio')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Calcular la disponibilidad promedio de 365 días por barrio y ordenarlo
mean_availability_by_neighbourhood = dataBerNoOutliers.groupby('neighbourhood_group_cleansed')['availability_365'].mean().reset_index()
mean_availability_by_neighbourhood = mean_availability_by_neighbourhood.sort_values(by='availability_365', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=mean_availability_by_neighbourhood, x='neighbourhood_group_cleansed', y='availability_365', ci=None)
plt.title('Disponibilidad promedio de 365 días por Barrio')
plt.xlabel('Barrio')
plt.ylabel('Disponibilidad promedio de 365 días')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# Análisis entre barrios y precio
plt.figure(figsize=(12, 6))
sns.boxplot(data=dataBerNoOutliers, x='neighbourhood_group_cleansed', y='price')
plt.title('Precio por Barrio')
plt.xlabel('Barrio')
plt.ylabel('Precio')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Análisis entre tipo de habitación y precio
plt.figure(figsize=(12, 6))
sns.boxplot(data=dataBerNoOutliers, x='room_type', y='price')
plt.title('Precio por Tipo de Habitación')
plt.xlabel('Tipo de Habitación')
plt.ylabel('Precio')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Análisis entre tipo de propiedad y precio
top_property_types = dataBerNoOutliers['property_type'].value_counts().head(10).index
filtered_data_property_type = dataBerNoOutliers[dataBerNoOutliers['property_type'].isin(top_property_types)]

plt.figure(figsize=(12, 6))
sns.boxplot(data=filtered_data_property_type, x='property_type', y='price')
plt.title('Precio por Tipo de Propiedad (Top 10)')
plt.xlabel('Tipo de Propiedad')
plt.ylabel('Precio')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Análisis entre barrios y disponibilidad de 365 días
plt.figure(figsize=(12, 6))
sns.boxplot(data=dataBerNoOutliers, x='neighbourhood_group_cleansed', y='availability_365')
plt.title('Disponibilidad de 365 días por Barrio')
plt.xlabel('Barrio')
plt.ylabel('Disponibilidad de 365 días')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


