import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

class GraphicalAnalyzer:
    def __init__(self, df: pd.DataFrame):
        """
        Инициализация анализатора с DataFrame
        
        Parameters:
        df (pd.DataFrame): DataFrame для анализа
        """
        self.df = df
        self.set_style()
    
    def set_style(self):
        """Установка стиля графиков"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def plot_correlation_heatmap(self, columns: Optional[List[str]] = None, 
                               annot: bool = True, cmap: str = 'coolwarm'):
        """
        Построение тепловой карты корреляции
        
        Parameters:
        columns (List[str]): Список столбцов для анализа (по умолчанию все числовые)
        annot (bool): Показывать значения корреляции на графике
        cmap (str): Цветовая карта
        """
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        
        if columns:
            valid_columns = [col for col in columns if col in numeric_columns]
            if not valid_columns:
                print("Нет подходящих числовых столбцов для анализа")
                return
            data_for_corr = self.df[valid_columns]
        else:
            data_for_corr = self.df[numeric_columns]
        
        if len(data_for_corr.columns) < 2:
            print("Недостаточно числовых столбцов для построения корреляции")
            return
        
        correlation_matrix = data_for_corr.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(correlation_matrix, 
                   annot=annot, 
                   cmap=cmap, 
                   center=0,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={"shrink": 0.8},
                   ax=ax)
        
        plt.title('Матрица корреляции', fontsize=16, pad=20)
        plt.tight_layout()
        plt.show()
        
        return correlation_matrix

    def plot_categorical_distribution(self, categorical_columns: Optional[List[str]] = None, 
                                    max_categories: int = 15, top_n: Optional[int] = None):
        """
        Построение столбчатых диаграмм для категориальных переменных
        
        Parameters:
        categorical_columns (List[str]): Список категориальных столбцов
        max_categories (int): Максимальное количество категорий для отображения
        top_n (int): Показать только top_n самых частых категорий
        """
        if categorical_columns is None:
            categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not categorical_columns:
            print("В данных нет категориальных столбцов")
            return
        
        n_cols = min(3, len(categorical_columns))
        n_rows = (len(categorical_columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, column in enumerate(categorical_columns):
            if i >= len(axes):
                break
                
            ax = axes[i]
            value_counts = self.df[column].value_counts()
            
            if top_n:
                value_counts = value_counts.head(top_n)
            elif len(value_counts) > max_categories:
                value_counts = value_counts.head(max_categories)
            
            if len(value_counts) == 0:
                ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{column}\n(нет данных)')
                continue
            
            bars = ax.bar(range(len(value_counts)), value_counts.values, color=sns

color_palette("husl", len(value_counts)))
            ax.set_title(f'Распределение: {column}', fontsize=14, pad=10)
            ax.set_xlabel(column)
            ax.set_ylabel('Количество')
            
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
            
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height}',
                       ha='center', va='bottom', fontsize=10)
        
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.show()

    def plot_pairplot(self, columns: Optional[List[str]] = None, hue: Optional[str] = None):
        """
        Построение pairplot для визуализации взаимосвязей
        
        Parameters:
        columns (List[str]): Список столбцов для анализа
        hue (str): Столбец для группировки по цвету
        """
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        
        if columns:
            valid_columns = [col for col in columns if col in self.df.columns]
            if not valid_columns:
                print("Указанные столбцы не найдены")
                return
            plot_data = self.df[valid_columns]
        else:
            plot_data = self.df[numeric_columns]
        
        if len(plot_data.columns) < 2:
            print("Недостаточно столбцов для построения pairplot")
            return
        
        if hue and hue not in plot_data.columns:
            plot_data[hue] = self.df[hue]
        
        sns.pairplot(plot_data, hue=hue, diag_kind='hist', corner=False)
        plt.suptitle('Pairplot: взаимосвязи между переменными', y=1.02)
        plt.show()

    def plot_distribution_comparison(self, numerical_column: str, categorical_column: str):
        """
        Сравнение распределений числовой переменной по категориям
        
        Parameters:
        numerical_column (str): Числовой столбец
        categorical_column (str): Категориальный столбец
        """
        if numerical_column not in self.df.columns or categorical_column not in self.df.columns:
            print("Указанные столбцы не найдены в данных")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        sns.boxplot(data=self.df, x=categorical_column, y=numerical_column, ax=ax1)
        ax1.set_title(f'Boxplot: {numerical_column} по {categorical_column}')
        ax1.tick_params(axis='x', rotation=45)
        
        sns.violinplot(data=self.df, x=categorical_column, y=numerical_column, ax=ax2)
        ax2.set_title(f'Violin plot: {numerical_column} по {categorical_column}')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

def analyze_dataset(df, categorical_cols=None, numerical_cols=None):
    """
    Функция для комплексного анализа датасета
    
    Parameters:
    df (pd.DataFrame): Датасет для анализа
    categorical_cols (List[str]): Список категориальных столбцов
    numerical_cols (List[str]): Список числовых столбцов
    """
    analyzer = GraphicalAnalyzer(df)
    
    print("=" * 50)
    print("АНАЛИЗ ДАТАСЕТА")
    print("=" * 50)
    print(f"Размер данных: {df.shape}")
    print(f"Количество строк: {df.shape[0]}")
    print(f"Количество столбцов: {df.shape[1]}")
    print("\nТипы данных:")
    print(df.dtypes.value_counts())
    
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if categorical_cols:
        print(f"\nКатегориальные столбцы ({len(categorical_cols)}): {categorical_cols}")
        analyzer.plot_categorical_distribution(categorical_cols)
    
    if numerical_cols:
        print(f"\nЧисловые столбцы ({len(numerical_cols)}): {numerical_cols}")



if len(numerical_cols) >= 2:
            print("\nМатрица корреляции:")
            analyzer.plot_correlation_heatmap(numerical_cols)
            
            if len(numerical_cols) <= 8:
                print("\nPairplot числовых переменных:")
                analyzer.plot_pairplot(numerical_cols)
        
        if categorical_cols and numerical_cols:
            print("\nСравнение распределений:")
            for num_col in numerical_cols[:2]:
                for cat_col in categorical_cols[:2]:
                    if len(df[cat_col].unique()) <= 10:
                        analyzer.plot_distribution_comparison(num_col, cat_col)
                        break

if name == "__main__":
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'age': np.random.randint(18, 65, 100),
        'salary': np.random.normal(50000, 15000, 100),
        'department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing'], 100),
        'experience': np.random.randint(0, 20, 100),
        'city': np.random.choice(['Moscow', 'SPb', 'Kazan', 'Novosibirsk'], 100),
        'satisfaction': np.random.randint(1, 6, 100)
    })
    
    analyze_dataset(sample_data)


