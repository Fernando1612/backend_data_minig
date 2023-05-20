import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
        if self.data is not None:
            return self.data.head(num_rows).to_json()
        else:
            print("No se han cargado los datos. Utiliza el método 'load_data()' primero.")
    
    def summary_statistics(self):
        if self.data is not None:
            return self.data.describe().to_json()
        else:
            print("No se han cargado los datos. Utiliza el método 'load_data()' primero.")
    
    def column_names(self):
        if self.data is not None:
            return self.data.columns.tolist()
        else:
            print("No se han cargado los datos. Utiliza el método 'load_data()' primero.")
            
    def missing_values(self):
        if self.data is not None:
            return self.data.isnull().sum()
        else:
            print("No se han cargado los datos. Utiliza el método 'load_data()' primero.")
            
    def plot_outliers_histogram(self, column):
        if self.data is not None:
            if column in self.data.columns:
                plt.figure(figsize=(8, 6))
                sns.histplot(data=self.data, x=column)
                plt.title(f"Histograma de '{column}'")
                plt.show()
            else:
                print(f"No se encuentra la columna '{column}' en los datos.")
        else:
            print("No se han cargado los datos. Utiliza el método 'load_data()' primero.")
    
    def plot_outliers_boxplot(self, column):
        if self.data is not None:
            if column in self.data.columns:
                plt.figure(figsize=(8, 6))
                sns.boxplot(data=self.data, y=column)
                plt.title(f"Diagrama de caja de '{column}'")
                plt.show()
            else:
                print(f"No se encuentra la columna '{column}' en los datos.")
        else:
            print("No se han cargado los datos. Utiliza el método 'load_data()' primero.")
    
    def plot_correlation_heatmap(self):
        if self.data is not None:
            correlation_matrix = self.data.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
            plt.title("Mapa de calor de correlación")
            plt.show()
        else:
            print("No se han cargado los datos. Utiliza el método 'load_data()' primero.")
