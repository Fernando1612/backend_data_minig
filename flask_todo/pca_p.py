import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Configurar el backend de Matplotlib antes de importarlo
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

class PCA_P:
    def __init__(self, n_components):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.data = None
        self.scaler = None
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
        
    def fit(self, scaler_type="StandardScaler"):
        if self.data is None:
            raise ValueError("No se han cargado los datos. Utilice el método 'load_data' para cargar los datos desde un archivo CSV.")
        
        # Preprocesar los datos eliminando columnas con valores de tipo string
        self.preprocess_data()
        
        # Separar las características de la matriz de datos
        X = self.data.iloc[:, :-1].values
        
        # Escoger el tipo de estandarización
        if scaler_type == "StandardScaler":
            self.scaler = StandardScaler()
        elif scaler_type == "MinMaxScaler":
            self.scaler = MinMaxScaler()
        elif scaler_type == "Normalizer":
            self.scaler = Normalizer()
        else:
            raise ValueError("Tipo de estandarización no válido. Los tipos válidos son: 'StandardScaler', 'MinMaxScaler', 'Normalizer'.")
        
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
    
    def plot_variance(self, n_components=2):
        if self.pca.explained_variance_ratio_ is None:
            raise ValueError("No se ha realizado el ajuste PCA. Utilice el método 'fit' antes de llamar a 'plot_variance'.")
        # Calcular la varianza acumulada
        cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
        varianza = sum(self.pca.explained_variance_ratio_[0:n_components])
        # Crear el gráfico de la varianza acumulada
        plt.plot(cumulative_variance)
        plt.xlabel('Número de componentes')
        plt.ylabel('Varianza acumulada')
        file_name = f"varianza.png"  # Genera un nombre de archivo único
        plt.grid()
        plt.savefig(file_name)  # Guarda la imagen en un archivo
        plt.close()  # Cierra la figura para liberar memoria
        return file_name, varianza

