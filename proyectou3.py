import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, classification_report
from wordcloud import WordCloud
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Ruta del archivo de entrada
data_path = r"C:\\Users\\alexi\\Desktop\\INTELIGENCIA NEGOCIOS\\ProyectoFinal\\encuesta.xlsx"

# =============================================
# 1. Carga y Preparación de los Datos
# =============================================

# Cargar el dataset
data = pd.read_excel(data_path)
data.columns = data.columns.str.strip()  # Eliminar espacios en nombres de columnas

# Verificar nombres de columnas
print("Nombres de columnas:", data.columns.tolist())

# Dividir columnas por tipo de dato
numeric_cols = [
    "Cómo calificarías el funcionamiento de TikTok?",
    "¿Cuántas veces al día abres la aplicación de TikTok?",
    "¿Cuántos videos compartes diariamente desde TikTok?",
    "Con cuántos amigos te compartes TikTok?",
    "Del 1 - 10 que tanto te gusta TikTok",
]
text_cols = [
    "¿Qué te motiva a usar TikTok diariamente?",
    "¿Qué tipo de contenido prefieres consumir en TikTok (educativo, entretenimiento, humor, tendencias, etc.)?",
    "¿Cómo describirías la influencia de TikTok en tu estado de ánimo diario?",
    "¿Qué cambios has notado en tu rutina diaria desde que empezaste a usar TikTok?",
    "¿Qué opinas del impacto que tiene TikTok en las relaciones sociales o personales?",
]

# Validar que las columnas existen en el dataset
numeric_cols = [col for col in numeric_cols if col in data.columns]
text_cols = [col for col in text_cols if col in data.columns]

if not numeric_cols:
    raise ValueError("No se encontraron columnas numéricas válidas en el archivo.")

if not text_cols:
    raise ValueError("No se encontraron columnas textuales válidas en el archivo.")

num_data = data[numeric_cols]
text_data = data[text_cols]

# Manejar valores faltantes
num_data = num_data.dropna()
text_data = text_data.fillna("N/A")

# Asegurar que las columnas de texto sean cadenas
text_data = text_data.astype(str)

# =============================================
# 2. Análisis Exploratorio de Datos
# =============================================

# 2.1 Resumen descriptivo
print("Resumen Estadístico:")
print(num_data.describe())

# 2.2 Visualización de datos numéricos
num_data.hist(bins=12, figsize=(10, 7), edgecolor='k')
plt.suptitle("Distribución de Variables Numéricas", fontsize=14)
plt.show()

sns.pairplot(num_data, diag_kind='kde', height=2)
plt.suptitle("Relaciones entre Variables Numéricas", y=1.02, fontsize=14)
plt.show()

# 2.3 Generar nube de palabras
total_text = " ".join(text_data.values.flatten())
wordcloud = WordCloud(width=800, height=400, background_color="black", colormap="coolwarm").generate(total_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Nube de Palabras", fontsize=16)
plt.show()

# =============================================
# 3. Clustering con K-Means
# =============================================

# Normalizar datos numéricos
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(num_data)

# Determinar el número de clústeres usando el método del codo
inertia_values = []
for k in range(2, 10):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(scaled_data)
    inertia_values.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(2, 10), inertia_values, marker='o')
plt.title("El Método del Codo", fontsize=14)
plt.xlabel("Número de Clústeres")
plt.ylabel("Inercia")
plt.grid()
plt.show()

# Ajustar KMeans con el número óptimo de clústeres
optimal_k = 4
kmeans_model = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_model.fit(scaled_data)
num_data['Cluster'] = kmeans_model.labels_

# Visualizar clústeres
sns.pairplot(num_data, hue="Cluster", diag_kind="kde", palette="Set1", height=2.5)
plt.suptitle("Clústeres Resultantes", y=1.02, fontsize=14)
plt.show()

# Características por clúster
for i in range(optimal_k):
    cluster_data = num_data[num_data['Cluster'] == i]
    print(f"Clúster {i} - Promedios de Características:")
    print(cluster_data.mean())

# =============================================
# 4. Clasificación con Naive Bayes
# =============================================

# Crear variable objetivo
num_data['Satisfaction'] = num_data["Del 1 - 10 que tanto te gusta TikTok"] > 5
X = scaled_data
y = num_data['Satisfaction']

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelo de Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Predicción y evaluación
y_pred = nb_model.predict(X_test)
print("Reporte de Clasificación (Naive Bayes):")
print(classification_report(y_test, y_pred))

# =============================================
# Finalización
# =============================================
print("Análisis completado. Guarda los resultados y sube el código actualizado al repositorio.")
