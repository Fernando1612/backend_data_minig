import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Configurar el backend de Matplotlib antes de importarlo
import matplotlib.pyplot as plt
import uuid

class EDA:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        
    def load_data(self):
        try:
            self.data = pd.read_csv(self.file_path)
            print("¡Datos cargados exitosamente!")
        except FileNotFoundError:
            print("Error al cargar el archivo. Verifica la ruta proporcionada.")
            
    def preview_data(self, num_rows=5):
        # Devuelve una vista previa de los datos (las primeras filas)
        if self.data is not None:
            preview = self.data.head(num_rows)
            preview = preview.fillna('Null')  # Agregar la palabra "Null" a los valores nulos
            return preview
        else:
            print("No se han cargado los datos. Utiliza el método 'load_data()' primero.")

    def summary_statistics(self):
        # Devuelve estadísticas resumidas de los datos (media, desviación estándar, etc.)
        if self.data is not None:
            return self.data.describe()
        else:
            print("No se han cargado los datos. Utiliza el método 'load_data()' primero.")
    
    def column_names(self):
        # Devuelve los nombres de las columnas de los datos
        if self.data is not None:
            return self.data.columns.tolist()
        else:
            print("No se han cargado los datos. Utiliza el método 'load_data()' primero.")
            
    def missing_values(self):
        # Devuelve la cantidad de valores faltantes en cada columna
        if self.data is not None:
            return self.data.isnull().sum().to_frame()
        else:
            print("No se han cargado los datos. Utiliza el método 'load_data()' primero.")

    def get_all_data(self):
        # Devuelve todos los datos después de eliminar las columnas con strings y las filas con valores nulos
        if self.data is not None:
            # Eliminar columnas con strings
            self.data = self.data.select_dtypes(exclude=['object'])
            # Eliminar filas con valores nulos
            self.data = self.data.dropna()
            return self.data
        else:
            print("No se han cargado los datos. Utiliza el método 'load_data()' primero.")
            
    def plot_outliers_histogram(self):
        # Genera un histograma de los datos para visualizar los posibles valores atípicos
        if self.data is not None:
            plt.figure(figsize=(8, 6))
            sns.histplot(data=self.data)
            plt.title("Histograma de datos")
            file_name = f"histogram.png"  # Genera un nombre de archivo único
            plt.savefig(file_name)  # Guarda la imagen en un archivo
            plt.close()  # Cierra la figura para liberar memoria
            return file_name
        else:
            print("No se han cargado los datos. Utiliza el método 'load_data()' primero.")
    
    def plot_outliers_boxplot(self):
        # Genera un diagrama de caja de los datos para visualizar los posibles valores atípicos
        if self.data is not None:
            sns.set(style="whitegrid")  # Establece el estilo de la cuadrícula
            plt.figure(figsize=(10, 6))  # Ajusta el tamaño de la figura
            sns.boxplot(data=self.data)
            plt.title("Diagrama de caja de datos")
            plt.xticks(rotation=90)  # Rota las etiquetas del eje x para una mejor legibilidad
            plt.tight_layout()  # Ajusta el espaciado entre los elementos de la figura
            file_name = f"boxplot.png"  # Genera un nombre de archivo único
            plt.savefig(file_name)  # Guarda la imagen en un archivo
            plt.close()  # Cierra la figura para liberar memoria
            return file_name
        else:
            print("No se han cargado los datos. Utiliza el método 'load_data()' primero.")
    
    def plot_correlation_heatmap(self):
        # Genera un mapa de calor de la correlación entre las variables numéricas de los datos
        if self.data is not None:
            self.data = self.data.select_dtypes(exclude=['object'])
            self.data = self.data.dropna()
            correlation_matrix = self.data.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
            plt.title("Mapa de calor de correlación")
            file_name = f"heatmap.png"  # Genera un nombre de archivo único
            plt.savefig(file_name)  # Guarda la imagen en un archivo
            plt.close()  # Cierra la figura para liberar memoria
            return file_name
        else:
            print("No se han cargado los datos. Utiliza el método 'load_data()' primero.")
