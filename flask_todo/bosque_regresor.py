import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')  # Configurar el backend de Matplotlib antes de importarlo
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from kneed import KneeLocator

class BosqueRegresor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def load_data(self):
        try:
            self.data = pd.read_csv(self.file_path)
            print("¡Datos cargados exitosamente!")
        except FileNotFoundError:
            print("Error al cargar el archivo. Verifica la ruta proporcionada.")

    def column_names(self, target=None):
        """
        Devuelve los nombres de las columnas numéricas del conjunto de datos.

        Args:
            target: Nombre de la columna objetivo a excluir (opcional).

        Returns:
            Una lista de nombres de columnas numéricas.
        """
        if self.data is not None:
            if target == None:
                numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
                return numeric_columns
            else:
                del self.data[target]
                numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
                return numeric_columns
        else:
            print("No se han cargado los datos. Utiliza el método 'load_data()' primero.")

    def cargar_datos(self, csv_file, target_column):
        """
        Carga los datos desde un archivo CSV y realiza el preprocesamiento básico.

        Args:
            csv_file: Ruta del archivo CSV.
            target_column: Nombre de la columna objetivo.

        """
        self.data = pd.read_csv(csv_file)
        self.data.dropna(inplace=True)  # Eliminar valores nulos
        self.X = self.data.drop(columns=[target_column])
        self.X = self.X.select_dtypes(exclude=['object'])
        self.y = self.data[target_column]

    def dividir_datos(self, test_size=0.2, random_state=42):
        """
        Divide los datos en conjuntos de entrenamiento y prueba.

        Args:
            test_size: Proporción del conjunto de datos a usar como datos de prueba.
            random_state: Semilla para la generación de números aleatorios.

        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, 
                                                                    test_size = 0.2, 
                                                                    random_state = 0, 
                                                                    shuffle = True)

    def entrenar_modelo(self, n_estimators=100, criterion='mse', max_depth=None,
                        min_samples_split=2, min_samples_leaf=1, max_features='auto'):
        """
        Entrena un modelo de RandomForestRegressor con los parámetros especificados.

        Args:
            n_estimators: Número de árboles en el bosque.
            criterion: Criterio utilizado para medir la calidad de una división.
            max_depth: Profundidad máxima de los árboles.
            min_samples_split: Número mínimo de muestras requeridas para dividir un nodo interno.
            min_samples_leaf: Número mínimo de muestras requeridas para estar en un nodo hoja.
            max_features: Número de características a considerar al buscar la mejor división.

        """
        self.model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion,
                                            max_depth=max_depth, min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf, max_features=max_features)
        self.model.fit(self.X_train, self.y_train)

    def predecir(self, X):
        """
        Realiza predicciones utilizando el modelo entrenado.

        Args:
            X: Datos de entrada.

        Returns:
            Un array con las predicciones.

        """
        self.X = self.X.select_dtypes(exclude=['object'])
        return self.model.predict(X)

    def evaluar_modelo(self):
        """
        Evalúa el modelo utilizando métricas de evaluación.

        Returns:
            mse: Error cuadrático medio.
            criterion: Criterio utilizado en el modelo.
            feature_importances: Importancia de las características.
            mae: Error absoluto medio.
            rmse: Error cuadrático medio.
            r2: Coeficiente de determinación R^2.

        """
        y_pred = self.predecir(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)
        feature_importances = self.model.feature_importances_
        criterion = self.model.criterion

        return mse, criterion, feature_importances, mae, rmse, r2
    
    def graficar_pronostico(self):
        """
        Grafica los resultados de las predicciones en comparación con los valores reales.

        """
        y_pred = self.predecir(self.X_test)
        plt.figure(figsize=(20, 5))
        plt.plot(np.array(self.y_test), color='red', marker='+', label='Real')
        plt.plot(y_pred, color='green', marker='+', label='Estimado')
        plt.title('Pronóstico')
        plt.grid(True)
        plt.legend()
        plt.savefig('pronostico.png')
        plt.close()

    def guardar_modelo(self, file_path):
        """
        Guarda el modelo entrenado en un archivo.

        Args:
            file_path: Ruta del archivo donde se guardará el modelo.

        """
        with open(file_path, 'wb') as file:
            pickle.dump(self.model, file)
        print("Modelo guardado exitosamente.")

    def cargar_modelo(self, file_path):
        """
        Carga un modelo guardado desde un archivo.

        Args:
            file_path: Ruta del archivo que contiene el modelo.

        """
        with open(file_path, 'rb') as file:
            self.model = pickle.load(file)
        print("Modelo cargado exitosamente.")
