import psycopg2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

try:
    conn = psycopg2.connect(
        port=5432,
        database="mediciones_calidad",
        user="postgres",
        password="0513"
    )
    print("Conexión exitosa")

    cur = conn.cursor()
    cur.execute("SELECT * FROM medicions")

    # Almacenar los resultados en una lista de tuplas
    results = []
    for row in cur.fetchall():
        results.append(row)

    # Crear un DataFrame de Pandas
    df = pd.DataFrame(results, columns=[desc[0] for desc in cur.description])

    # Convertir la columna 'Ubicacion' a minúsculas
    df['Ubicacion'] = df['Ubicacion'].str.lower()

    pd.set_option('display.max_columns', None)
    
    # Mostrar los primeros 5 datos del DataFrame
    print(df.head())

    cur.close()
    conn.close()

except Exception as error:
    print("Error:", error)

# Rangos de aceptación para cada parámetro
rangos_aceptacion = {
    'pH': (5.5, 7.5),
    'conductividad': (0.5, 3),
    'nitratos': (5, 20),
    'nivel': (5, 45)
}

# Crear una nueva columna "parametro_cumple" e inicializarla como True
df['parametro_cumple'] = True

# Iterar sobre los parámetros y actualizar la columna "parametro_cumple" según si cumple o no
for parametro, rango in rangos_aceptacion.items():
    df[parametro + '_cumple'] = (df['Parametro'] == parametro) & (df['Valor'] >= rango[0]) & (df['Valor'] <= rango[1])

# Actualizar la columna "parametro_cumple" basada en las columnas de cumplimiento individuales
df['parametro_cumple'] = df[[parametro + '_cumple' for parametro in rangos_aceptacion.keys()]].any(axis=1)

pd.set_option('display.max_columns', None)

# Convertir los valores booleanos a enteros (1 para True, 0 para False)
df['parametro_cumple'] = df['parametro_cumple'].astype(int)
df['pH_cumple'] = df['pH_cumple'].astype(int)
df['conductividad_cumple'] = df['conductividad_cumple'].astype(int)
df['nitratos_cumple'] = df['nitratos_cumple'].astype(int)
df['nivel_cumple'] = df['nivel_cumple'].astype(int)

# Mostrar el DataFrame resultante
print(df)

# Filtrar las filas para cada parámetro y seleccionar las columnas 'Valor'
pH_values = df.loc[df['Parametro'] == 'pH', 'Valor']
conductividad_values = df.loc[df['Parametro'] == 'conductividad', 'Valor']
nitratos_values = df.loc[df['Parametro'] == 'nitratos', 'Valor']
nivel_values = df.loc[df['Parametro'] == 'nivel', 'Valor']

# Mostrar una muestra de los valores
print("Valores de pH:")
print(pH_values.head())

print("\nValores de conductividad:")
print(conductividad_values.head())

print("\nValores de nitratos:")
print(nitratos_values.head())

print("\nValores de nivel:")
print(nivel_values.head())

# Seleccionar las filas correspondientes a los parámetros de interés ('pH', 'conductividad', 'nitratos' y 'nivel')
caracteristicas = df[df['Parametro'].isin(['pH', 'conductividad', 'nitratos', 'nivel'])]

# Crear un nuevo DataFrame con solo las columnas de Valor y Parametro
caracteristicas = caracteristicas[['Valor', 'Parametro']]

# Reorganizar el DataFrame para tener cada parámetro como una columna
caracteristicas = caracteristicas.pivot(columns='Parametro', values='Valor')

# Reemplazar NaN con 0 en el DataFrame de características
caracteristicas = caracteristicas.fillna(0)

# Mostrar las primeras filas del conjunto de características
print(caracteristicas.head())

# Seleccionar la etiqueta
etiquetas = df['parametro_cumple']

# Imprimir una muestra de las etiquetas
print("\nMuestra de etiquetas:")
print(etiquetas.head())

# Dividir los datos en conjuntos de entrenamiento y prueba
X = caracteristicas[['conductividad', 'nitratos', 'nivel', 'pH']]
y = df['parametro_cumple']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo SVM
model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred_train = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred_train)
print("Precision en el conjunto de entrenamiento:", train_accuracy)

y_pred_test = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print("Precision en el conjunto de prueba:", test_accuracy)
