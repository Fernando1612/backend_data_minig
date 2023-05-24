import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class PCA_P:
    def __init__(self, n_components):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.data = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self, filename):
        # Leer el archivo CSV
        self.data = pd.read_csv(filename)
        
    def preprocess_data(self):
        if self.data is None:
            raise ValueError("No se han cargado los datos. Utilice el método 'load_data' para cargar los datos desde un archivo CSV.")
        
        # Identificar columnas con valores de tipo string
        string_columns = self.data.select_dtypes(include=['object']).columns
        
        # Eliminar las columnas con valores de tipo string
        self.data = self.data.drop(string_columns, axis=1)
        
        # Eliminar las filas con valores NaN
        self.data = self.data.dropna()

        # Guardar los nombres de las características
        self.feature_names = list(self.data.columns)[:-1]
        
    def fit(self):
        if self.data is None:
            raise ValueError("No se han cargado los datos. Utilice el método 'load_data' para cargar los datos desde un archivo CSV.")
        
        # Preprocesar los datos eliminando columnas con valores de tipo string
        self.preprocess_data()
        
        # Separar las características de la matriz de datos
        X = self.data.iloc[:, :-1].values
        
        # Estandarizar los datos
        X_std = self.scaler.fit_transform(X)
        
        # Ajustar el modelo PCA a los datos estandarizados
        self.pca.fit(X_std)
        
    def transform(self):
        if self.data is None:
            raise ValueError("No se han cargado los datos. Utilice el método 'load_data' para cargar los datos desde un archivo CSV.")
        
        # Preprocesar los datos eliminando columnas con valores de tipo string
        self.preprocess_data()
        
        # Separar las características de la matriz de datos
        X = self.data.iloc[:, :-1].values
        
        # Estandarizar los datos
        X_std = self.scaler.transform(X)
        
        # Aplicar la transformación PCA a los datos estandarizados
        X_transformed = self.pca.transform(X_std)
        
        # Crear el DataFrame transformado con los nombres de columna adecuados
        transformed_df = pd.DataFrame(X_transformed, columns=['Component ' + str(i+1) for i in range(self.n_components)])
        
        # Agregar la columna de etiquetas si está presente en el DataFrame original
        if 'Label' in self.data.columns:
            transformed_df['Label'] = self.data['Label'].values
        
        return transformed_df
