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
dataBcn = pd.read_csv("CityFiles/barcelona/listings.csv")
dataBaa = pd.read_csv("CityFiles/buenos_aires/listings.csv")


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

def preprocess_data(data):
    # #Eliminar columnas
    cleaned_data = data.drop(columns=columns_to_drop, axis=1, inplace=False)

    # Transformar dades
    cleaned_data['price'] = cleaned_data['price'].str.replace('$', '', regex=False)
    cleaned_data['price'] = cleaned_data['price'].str.replace(',', '', regex=False)
    cleaned_data['price'] = cleaned_data['price'].astype(float)
    mean_price = cleaned_data['price'].mean()
    cleaned_data['price'].fillna(mean_price, inplace=True)
    mean_review_scores_rating = cleaned_data['review_scores_rating'].mean()
    cleaned_data['review_scores_rating'].fillna(mean_review_scores_rating, inplace=True)
    return cleaned_data

dataBerCleaned = preprocess_data(dataBer)
dataBcnCleaned = preprocess_data(dataBcn)
dataBaaCleaned = preprocess_data(dataBaa)

"""
|||||||||||||||||||||||||||||||||||||||DISTRIBUCIÓN POR BARRIO |||||||||||||||||||||||||||||||||||||||||||
"""


def analysis_price_neighbourhood(data, group):
    if group:
        column_neigh = 'neighbourhood_group_cleansed'
    else:
        column_neigh = 'neighbourhood_cleansed'
    
    # Frecuencia por barrio (cantidad de listados)
    freq_by_neighbourhood = data[column_neigh].value_counts().reset_index()
    freq_by_neighbourhood.columns = [column_neigh, 'frequency']
    freq_by_neighbourhood = freq_by_neighbourhood.sort_values(by='frequency', ascending=False)

    # Precio medio por barrio
    mean_price_by_neighbourhood = data.groupby(column_neigh)['price'].mean().reset_index()
    mean_price_by_neighbourhood = mean_price_by_neighbourhood.sort_values(by='price', ascending=False)

    # Puntuación media por barrio (suponiendo que hay una columna 'review_scores_rating')
    mean_score_by_neighbourhood = data.groupby(column_neigh)['review_scores_rating'].mean().reset_index()
    mean_score_by_neighbourhood = mean_score_by_neighbourhood.sort_values(by='review_scores_rating', ascending=False)

    total_neighbourhoods = data[column_neigh].nunique()
    x = int(total_neighbourhoods*0.25)

    # Extraer los dos barrios con precios más altos y más bajos
    top_price = mean_price_by_neighbourhood.head(x)
    bottom_price = mean_price_by_neighbourhood.tail(x)
    mean_top_price = top_price['price'].mean()
    mean_bottom_price = bottom_price['price'].mean()
    percent_diff_price = ((mean_top_price - mean_bottom_price) / mean_bottom_price) * 100

    # Extraer los dos barrios con mayor y menor frecuencia
    top_freq = freq_by_neighbourhood.head(x)
    bottom_freq = freq_by_neighbourhood.tail(x)
    mean_top_freq = top_freq['frequency'].mean()
    mean_bottom_freq = bottom_freq['frequency'].mean()
    percent_diff_freq = ((mean_top_freq - mean_bottom_freq) / mean_bottom_freq) * 100

    # Extraer los dos barrios con puntuaciones más altas y más bajas
    top_score = mean_score_by_neighbourhood.head(x)
    bottom_score = mean_score_by_neighbourhood.tail(x)
    mean_top_score = top_score['review_scores_rating'].mean()
    mean_bottom_score = bottom_score['review_scores_rating'].mean()
    percent_diff_score = ((mean_top_score - mean_bottom_score) / mean_bottom_score) * 100


    # Imprimir resultados
    print("=== Frecuencia por Barrio ===")
    print(freq_by_neighbourhood)
    print(f"\nDiferencia porcentual en frecuencia entre los barrios con mayor y menor frecuencia: {percent_diff_freq:.2f}%\n")

    print("=== Precio promedio por Barrio ===")
    print(mean_price_by_neighbourhood)
    print(f"\nPrecio medio del 25% de los barrios más caros: {mean_top_price}")
    print(f"Precio medio del 25% de los barrios más baratos: {mean_bottom_price}")
    print(f"Diferencia porcentual en precio: {percent_diff_price:.2f}%\n")

    print("=== Puntuación media por Barrio ===")
    print(mean_score_by_neighbourhood)
    print(f"\nDiferencia porcentual en puntuación entre los barrios con mayor y menor puntuación: {percent_diff_score:.2f}%\n")

    # Gráfico de precio promedio por barrio
    plt.figure(figsize=(12, 6))
    sns.barplot(data=mean_price_by_neighbourhood, x=column_neigh, y='price', ci=None)
    plt.title('Precio promedio por Barrio')
    plt.xlabel('Barrio')
    plt.ylabel('Precio promedio')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Gráfico de frecuencia por barrio
    plt.figure(figsize=(12, 6))
    sns.barplot(data=freq_by_neighbourhood, x=column_neigh, y='frequency', ci=None)
    plt.title('Frecuencia por Barrio')
    plt.xlabel('Barrio')
    plt.ylabel('Frecuencia')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Gráfico de puntuación promedio por barrio
    plt.figure(figsize=(12, 6))
    sns.barplot(data=mean_score_by_neighbourhood, x=column_neigh, y='review_scores_rating', ci=None)
    plt.title('Puntuación promedio por Barrio')
    plt.xlabel('Barrio')
    plt.ylabel('Puntuación promedio')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

"""
 ----------> BERLIN
"""
analysis_price_neighbourhood(dataBerCleaned, True)
analysis_price_neighbourhood(dataBcnCleaned, True)
analysis_price_neighbourhood(dataBaaCleaned, False)


"""
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



"""