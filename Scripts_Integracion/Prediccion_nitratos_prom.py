import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import statistics as stats
import os
import pandas as pd
import joblib
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import KNNImputer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

nRowsRead = 1000  # specify 'None' if want to read whole file
# data.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
file_url = 'https://raw.githubusercontent.com/luicamongi/Anexo_Magdalena/main/Anexo_nitratos_prom.csv'
df = pd.read_csv(file_url, delimiter=',', nrows=nRowsRead)
df.dataframeName = 'data.csv'
nRow, nCol = df.shape
print(f'There are {nRow} rows and {nCol} columns')

print(df.head(5))
print(df.isna().sum())
print(df)

columns_to_drop = [0, 1, 2, 3, 6, 8, 9, 10]
Z = df.drop(df.columns[columns_to_drop], axis=1)

W = Z.dropna()
print(W)

print(W.isna().sum())

# Calcular la matriz de correlación
corr_matrix = W.corr(numeric_only=True)

# Mostrar la matriz de correlación
print("Matriz de correlación:")
print(corr_matrix)

# Graficar la matriz de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de Correlación')
plt.show()

print(W.describe())

columns_to_drop_1 = [2, 3, 6, 8, 9, 10]
P = df.drop(df.columns[columns_to_drop_1], axis=1)

# OJO:
# En el notebook estaba Q = P.dropna(), pero así la imputación de nitratos casi no sirve.
# Se deja copia del DataFrame y después se imputan nitratos primero.
Q = P.copy()
print(Q)

# Imputación de valores faltantes en Nitratos
mean_values = Q.groupby(['Nombre punto de monitoreo'])['Nitratos (mg N-NO3-/L)'].transform('mean')
Q['Nitratos (mg N-NO3-/L)'] = Q['Nitratos (mg N-NO3-/L)'].fillna(mean_values)

if Q['Nitratos (mg N-NO3-/L)'].isna().sum() > 0:
    global_mean = Q['Nitratos (mg N-NO3-/L)'].mean()
    Q['Nitratos (mg N-NO3-/L)'] = Q['Nitratos (mg N-NO3-/L)'].fillna(global_mean)

# Eliminar filas con faltantes en variables clave
Q = Q.dropna(subset=['Temperatura °C', 'Conductividad Eléctrica uS/cm', 'pH unidades de pH']).copy()

# Definir las características (X) y la etiqueta (y)
X = Q[['Temperatura °C', 'Conductividad Eléctrica uS/cm', 'pH unidades de pH']]
y = Q['Nitratos (mg N-NO3-/L)']

# Agregar características polinómicas
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Normalizar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Definir el modelo Random Forest
rf_model = RandomForestRegressor(random_state=42)

# Definir la cuadrícula de hiperparámetros para Random Forest
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Configurar la búsqueda en cuadrícula para Random Forest
rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=5, n_jobs=-1, verbose=2, scoring='r2')

# Entrenar el modelo Random Forest
rf_grid_search.fit(X_train, y_train)

# Mejor conjunto de hiperparámetros para Random Forest
rf_best_params = rf_grid_search.best_params_
print("Mejores hiperparámetros para Random Forest:", rf_best_params)

# Evaluar el modelo Random Forest con los mejores hiperparámetros
rf_best_model = rf_grid_search.best_estimator_
rf_y_pred = rf_best_model.predict(X_test)

# Calcular el error cuadrático medio (MSE), el MAE y el coeficiente de determinación (R^2) para Random Forest
rf_mse = mean_squared_error(y_test, rf_y_pred)
rf_mae = mean_absolute_error(y_test, rf_y_pred)
rf_r2 = r2_score(y_test, rf_y_pred)

# Imprimir los resultados para Random Forest
print("Random Forest - Error cuadrático medio (MSE):", rf_mse)
print("Random Forest - Error Absoluto Medio (MAE):", rf_mae)
print("Random Forest - Coeficiente de determinación (R^2):", rf_r2)

# Definir el modelo Gradient Boosting
gb_model = GradientBoostingRegressor(random_state=42)

# Definir la cuadrícula de hiperparámetros para Gradient Boosting
gb_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Configurar la búsqueda en cuadrícula para Gradient Boosting
gb_grid_search = GridSearchCV(estimator=gb_model, param_grid=gb_param_grid, cv=5, n_jobs=-1, verbose=2, scoring='r2')

# Entrenar el modelo Gradient Boosting
gb_grid_search.fit(X_train, y_train)

# Mejor conjunto de hiperparámetros para Gradient Boosting
gb_best_params = gb_grid_search.best_params_
print("Mejores hiperparámetros para Gradient Boosting:", gb_best_params)

# Evaluar el modelo Gradient Boosting con los mejores hiperparámetros
gb_best_model = gb_grid_search.best_estimator_
gb_y_pred = gb_best_model.predict(X_test)

# Calcular el error cuadrático medio (MSE), el MAE y el coeficiente de determinación (R^2) para Gradient Boosting
gb_mse = mean_squared_error(y_test, gb_y_pred)
gb_mae = mean_absolute_error(y_test, gb_y_pred)
gb_r2 = r2_score(y_test, gb_y_pred)

# Imprimir los resultados para Gradient Boosting
print("Gradient Boosting - Error cuadrático medio (MSE):", gb_mse)
print("Gradient Boosting - Error Absoluto Medio (MAE):", gb_mae)
print("Gradient Boosting - Coeficiente de determinación (R^2):", gb_r2)

# Definir el modelo XGBoost
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Definir la cuadrícula de hiperparámetros para XGBoost
xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Configurar la búsqueda aleatoria para XGBoost
xgb_random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=xgb_param_grid,
    n_iter=50,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring='r2',
    random_state=42
)

# Entrenar el modelo XGBoost
xgb_random_search.fit(X_train, y_train)

# Mejor conjunto de hiperparámetros para XGBoost
xgb_best_params = xgb_random_search.best_params_
print("Mejores hiperparámetros para XGBoost:", xgb_best_params)

# Evaluar el modelo XGBoost con los mejores hiperparámetros
xgb_best_model = xgb_random_search.best_estimator_
xgb_y_pred = xgb_best_model.predict(X_test)

# Calcular el error cuadrático medio (MSE), el MAE y el coeficiente de determinación (R^2) para XGBoost
xgb_mse = mean_squared_error(y_test, xgb_y_pred)
xgb_mae = mean_absolute_error(y_test, xgb_y_pred)
xgb_r2 = r2_score(y_test, xgb_y_pred)

# Imprimir los resultados para XGBoost
print("XGBoost - Error cuadrático medio (MSE):", xgb_mse)
print("XGBoost - Error Absoluto Medio (MAE):", xgb_mae)
print("XGBoost - Coeficiente de determinación (R^2):", xgb_r2)

# Graficar las predicciones vs los valores reales para el mejor modelo
best_model = xgb_best_model  # Se deja XGBoost por el MAE

y_pred = best_model.predict(X_test)
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='red')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Mejor Modelo - Predicciones vs Valores Reales')
plt.legend(['Línea de identidad (y=x)', 'Predicciones'])
plt.show()

# Evaluar el mejor modelo usando validación cruzada
cv_scores = cross_val_score(best_model, X_scaled, y, cv=5, scoring='r2')
print("Puntajes R^2 de validación cruzada:", cv_scores)
print("Media de los puntajes R^2:", np.mean(cv_scores))

# Guardar transformadores y modelo final en la misma carpeta del script
joblib.dump(poly, os.path.join(BASE_DIR, "poly_nitratos.pkl"))
joblib.dump(scaler, os.path.join(BASE_DIR, "scaler_nitratos.pkl"))
joblib.dump(xgb_best_model, os.path.join(BASE_DIR, "modelo_xgb_nitratos.pkl"))

print("Se guardaron los archivos:")
print("poly_nitratos.pkl")
print("scaler_nitratos.pkl")
print("modelo_xgb_nitratos.pkl")

# Obtener la importancia de las características del modelo XGBoost
feature_importances = xgb_best_model.feature_importances_

# Crear un DataFrame para organizar las importancias junto con el nombre de las características
features = poly.get_feature_names_out(input_features=X.columns)
feature_importance_df = pd.DataFrame({
    'Característica': features,
    'Importancia': feature_importances
})

# Ordenar por importancia
feature_importance_df = feature_importance_df.sort_values(by='Importancia', ascending=False)

# Imprimir la importancia de las características
print(feature_importance_df)

# Graficar la importancia de las características
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Característica'], feature_importance_df['Importancia'])
plt.xlabel('Importancia')
plt.ylabel('Característica')
plt.title('Importancia de las Características en XGBoost')
plt.gca().invert_yaxis()  # Invertir el eje y para que la característica más importante aparezca arriba
plt.show()