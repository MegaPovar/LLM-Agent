import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class StatAgent:
    """
    Агент статистического анализа для работы в агентной системе.
    Работает через function calling для выполнения статистических тестов.
    """
    
    def __init__(self):
        self.dataset = None
        self.test_results = []
        self.significant_results = []
        
    def receive_dataset(self, data: pd.DataFrame):
        """Получить датасет от другого агента"""
        self.dataset = data
        print(f"Агент статистического анализа получил датасет: {data.shape[0]} строк, {data.shape[1]} колонок")
        return {"status": "success", "message": f"Датасет получен ({data.shape[0]}x{data.shape[1]})"}
    
    def analyze_dataset_structure(self):
        """Проанализировать структуру датасета для определения возможных тестов"""
        if self.dataset is None:
            return {"error": "Датасет не загружен"}
        
        column_types = {}
        for col in self.dataset.columns:
            dtype = str(self.dataset[col].dtype)
            unique_count = self.dataset[col].nunique()
            is_numeric = dtype.startswith(('int', 'float'))
            
            column_types[col] = {
                'dtype': dtype,
                'unique_count': unique_count,
                'is_numeric': is_numeric,
                'is_binary': is_numeric and unique_count == 2,
                'is_categorical': not is_numeric and unique_count <= 20
            }
        
        self.column_info = column_types
        return column_types
    
    def function_calling(self, function_name: str, **kwargs):
        """
        Система function calling для выполнения статистических тестов
        """
        function_map = {
            't_test_independent': self.perform_t_test_independent,
            't_test_paired': self.perform_t_test_paired,
            'chi_square_test': self.perform_chi_square_test,
            'pearson_correlation': self.perform_pearson_correlation,
            'spearman_correlation': self.perform_spearman_correlation,
            'auto_run_tests': self.auto_run_statistical_tests,
            'get_significant_results': self.get_significant_results
        }
        
        if function_name in function_map:
            result = function_map[function_name](**kwargs)
            
            # Сохраняем результаты тестов для последующего анализа
            if function_name not in ['auto_run_tests', 'get_significant_results']:
                self.test_results.append({
                    'test': function_name,
                    'parameters': kwargs,
                    'result': result
                })
                
                # Проверяем статистическую значимость
                if 'p_value' in result and result['p_value'] < 0.05:
                    self.significant_results.append({
                        'test': function_name,
                        'parameters': kwargs,
                        'result': result
                    })
            
            return result
        else:
            return {"error": f"Функция {function_name} не найдена", "available_functions": list(function_map.keys())}
    
    def perform_t_test_independent(self, column1: str, column2: str = None, group_column: str = None, group1: Any = None, group2: Any = None):
        """Независимый t-тест для сравнения двух групп"""
        if self.dataset is None:
            return {"error": "Датасет не загружен"}
        
        try:
            if column2:  # Сравнение двух колонок
                data1 = self.dataset[column1].dropna()
                data2 = self.dataset[column2].dropna()
                t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
                test_type = "Двухвыборочный t-тест (независимый)"
                
                result = {
                    "test": test_type,
                    "columns": [column1, column2],
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                    "n1": len(data1),
                    "n2": len(data2),
                    "mean1": float(data1.mean()),
                    "mean2": float(data2.mean())
                }
                
            elif group_column:  # Сравнение групп по одной колонке
                if group1 is None or group2 is None:
                    groups = self.dataset[group_column].dropna().unique()
                    if len(groups) >= 2:
                        group1, group2 = groups[:2]
                    else:
                        return {"error": "Недостаточно групп для сравнения"}
                
                data1 = self.dataset[self.dataset[group_column] == group1][column1].dropna()
                data2 = self.dataset[self.dataset[group_column] == group2][column1].dropna()
                
                t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
                test_type = "Двухвыборочный t-тест по группам"
                
                result = {
                    "test": test_type,
                    "column": column1,
                    "group_column": group_column,
                    "groups": [str(group1), str(group2)],
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                    "n1": len(data1),
                    "n2": len(data2),
                    "mean1": float(data1.mean()),
                    "mean2": float(data2.mean())
                }
            else:
                return {"error": "Недостаточно параметров для t-теста"}
            
            return result
            
        except Exception as e:
            return {"error": f"Ошибка при выполнении t-теста: {str(e)}"}
    
    def perform_t_test_paired(self, column1: str, column2: str):
        """Парный t-тест для зависимых выборок"""
        if self.dataset is None:
            return {"error": "Датасет не загружен"}
        
        try:
            data1 = self.dataset[column1].dropna()
            data2 = self.dataset[column2].dropna()
            
            # Выравниваем размеры выборок
            min_len = min(len(data1), len(data2))
            data1 = data1.iloc[:min_len]
            data2 = data2.iloc[:min_len]
            
            t_stat, p_value = stats.ttest_rel(data1, data2)
            
            result = {
                "test": "Парный t-тест",
                "columns": [column1, column2],
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "n_pairs": min_len,
                "mean_diff": float((data1 - data2).mean()),
                "std_diff": float((data1 - data2).std())
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Ошибка при выполнении парного t-теста: {str(e)}"}
    
    def perform_chi_square_test(self, column1: str, column2: str):
        """Тест хи-квадрат для категориальных переменных"""
        if self.dataset is None:
            return {"error": "Датасет не загружен"}
        
        try:
            # Создаем таблицу сопряженности
            contingency_table = pd.crosstab(self.dataset[column1], self.dataset[column2])
            
            if contingency_table.size == 0:
                return {"error": "Не удалось создать таблицу сопряженности"}
            
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            
            result = {
                "test": "Тест хи-квадрат",
                "columns": [column1, column2],
                "chi2_statistic": float(chi2),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "degrees_of_freedom": int(dof),
                "contingency_table_shape": contingency_table.shape
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Ошибка при выполнении теста хи-квадрат: {str(e)}"}
    
    def perform_pearson_correlation(self, column1: str, column2: str):
        """Корреляция Пирсона для двух числовых переменных"""
        if self.dataset is None:
            return {"error": "Датасет не загружен"}
        
        try:
            data1 = self.dataset[column1].dropna()
            data2 = self.dataset[column2].dropna()
            
            # Выравниваем размеры выборок
            min_len = min(len(data1), len(data2))
            data1 = data1.iloc[:min_len]
            data2 = data2.iloc[:min_len]
            
            corr_coef, p_value = stats.pearsonr(data1, data2)
            
            result = {
                "test": "Корреляция Пирсона",
                "columns": [column1, column2],
                "correlation_coefficient": float(corr_coef),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "n": min_len,
                "interpretation": self._interpret_correlation(corr_coef)
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Ошибка при выполнении корреляции Пирсона: {str(e)}"}
    
    def perform_spearman_correlation(self, column1: str, column2: str):
        """Ранговая корреляция Спирмена"""
        if self.dataset is None:
            return {"error": "Датасет не загружен"}
        
        try:
            data1 = self.dataset[column1].dropna()
            data2 = self.dataset[column2].dropna()
            
            # Выравниваем размеры выборок
            min_len = min(len(data1), len(data2))
            data1 = data1.iloc[:min_len]
            data2 = data2.iloc[:min_len]
            
            corr_coef, p_value = stats.spearmanr(data1, data2)
            
            result = {
                "test": "Корреляция Спирмена",
                "columns": [column1, column2],
                "correlation_coefficient": float(corr_coef),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "n": min_len,
                "interpretation": self._interpret_correlation(corr_coef)
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Ошибка при выполнении корреляции Спирмена: {str(e)}"}
    
    def _interpret_correlation(self, r):
        """Интерпретация коэффициента корреляции"""
        r_abs = abs(r)
        if r_abs >= 0.9:
            return "Очень сильная корреляция"
        elif r_abs >= 0.7:
            return "Сильная корреляция"
        elif r_abs >= 0.5:
            return "Умеренная корреляция"
        elif r_abs >= 0.3:
            return "Слабая корреляция"
        else:
            return "Очень слабая или отсутствующая корреляция"
    
    def identify_potential_tests(self):
        """Автоматически определить потенциальные тесты для датасета"""
        if self.dataset is None:
            return {"error": "Датасет не загружен"}
        
        self.analyze_dataset_structure()
        potential_tests = []
        
        numeric_cols = [col for col, info in self.column_info.items() if info['is_numeric']]
        binary_cols = [col for col, info in self.column_info.items() if info['is_binary']]
        categorical_cols = [col for col, info in self.column_info.items() if info['is_categorical']]
        
        # Генерация потенциальных пар для корреляций
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                potential_tests.append({
                    'test': 'pearson_correlation',
                    'params': {'column1': numeric_cols[i], 'column2': numeric_cols[j]}
                })
                potential_tests.append({
                    'test': 'spearman_correlation', 
                    'params': {'column1': numeric_cols[i], 'column2': numeric_cols[j]}
                })
        
        # Генерация пар для t-тестов
        for num_col in numeric_cols[:5]:  # Ограничиваем количество
            for cat_col in categorical_cols[:3]:
                potential_tests.append({
                    'test': 't_test_independent',
                    'params': {'column1': num_col, 'group_column': cat_col}
                })
        
        # Генерация пар для хи-квадрат
        for i in range(len(categorical_cols)):
            for j in range(i+1, len(categorical_cols)):
                if i < 3 and j < 3:  # Ограничиваем
                    potential_tests.append({
                        'test': 'chi_square_test',
                        'params': {'column1': categorical_cols[i], 'column2': categorical_cols[j]}
                    })
        
        return potential_tests[:30]  # Ограничиваем 30 тестами
    
    def auto_run_statistical_tests(self, max_tests: int = 30):
        """Автоматический запуск статистических тестов"""
        if self.dataset is None:
            return {"error": "Датасет не загружен"}
        
        potential_tests = self.identify_potential_tests()
        executed_tests = []
        
        print(f"Автоматический запуск {min(len(potential_tests), max_tests)} статистических тестов...")
        
        for i, test_info in enumerate(potential_tests[:max_tests]):
            try:
                print(f"Выполнение теста {i+1}: {test_info['test']}...")
                result = self.function_calling(test_info['test'], **test_info['params'])
                
                if 'error' not in result:
                    executed_tests.append({
                        'test_number': i+1,
                        'test_type': test_info['test'],
                        'params': test_info['params'],
                        'result': result
                    })
                    
                    if result.get('significant', False):
                        print(f"  Найдена статистически значимая зависимость!")
            except Exception as e:
                print(f"  Ошибка при выполнении теста: {str(e)}")
                continue
        
        print(f"\nВыполнено {len(executed_tests)} тестов")
        print(f"Найдено {len(self.significant_results)} статистически значимых результатов")
        
        return {
            "total_tests_executed": len(executed_tests),
            "significant_tests": len(self.significant_results),
            "test_details": executed_tests
        }
    
    def get_significant_results(self):
        """Получить статистически значимые результаты в формате строки для передачи другому агенту"""
        if not self.significant_results:
            return "STAT_RESULTS: Нет статистически значимых результатов (p < 0.05)"
        
        result_lines = ["STAT_RESULTS: Статистически значимые зависимости (p < 0.05):"]
        
        for i, res in enumerate(self.significant_results, 1):
            test_type = res['test']
            params = res['parameters']
            result = res['result']
            
            if test_type == 'pearson_correlation' or test_type == 'spearman_correlation':
                line = (f"{i}. {test_type.replace('_', ' ').title()}: "
                       f"{params['column1']} ↔ {params['column2']} | "
                       f"r = {result['correlation_coefficient']:.3f}, "
                       f"p = {result['p_value']:.4f}")
            
            elif 't_test' in test_type:
                if 'group_column' in params:
                    line = (f"{i}. {test_type.replace('_', ' ').title()}: "
                           f"{params['column1']} по группам {params['group_column']} | "
                           f"t = {result['t_statistic']:.3f}, "
                           f"p = {result['p_value']:.4f}")
                else:
                    line = (f"{i}. {test_type.replace('_', ' ').title()}: "
                           f"{params.get('column1', 'N/A')} ↔ {params.get('column2', 'N/A')} | "
                           f"t = {result['t_statistic']:.3f}, "
                           f"p = {result['p_value']:.4f}")
            
            elif test_type == 'chi_square_test':
                line = (f"{i}. Chi-square test: "
                       f"{params['column1']} ↔ {params['column2']} | "
                       f"χ² = {result['chi2_statistic']:.3f}, "
                       f"p = {result['p_value']:.4f}")
            
            else:
                line = f"{i}. {test_type}: p = {result.get('p_value', 'N/A'):.4f}"
            
            result_lines.append(line)
        
        return "\n".join(result_lines)


# Пример использования агента
def main():
    # Создаем тестовый датасет для демонстрации
    np.random.seed(42)
    n_samples = 100
    
    test_data = pd.DataFrame({
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'score_test1': np.random.normal(75, 15, n_samples),
        'score_test2': np.random.normal(78, 12, n_samples) + np.random.randn(n_samples) * 5,
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'satisfaction': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'purchased': np.random.choice([0, 1], n_samples),
        'hours_studied': np.random.exponential(10, n_samples),
        'exam_result': np.random.normal(70, 20, n_samples)
    })
    
    # Добавляем некоторые зависимости для демонстрации
    test_data['score_test2'] = test_data['score_test1'] * 0.7 + np.random.randn(n_samples) * 10
    test_data.loc[test_data['gender'] == 'Male', 'income'] += 5000
    test_data.loc[test_data['education'] == 'PhD', 'income'] += 10000
    
    # Создаем и настраиваем агента
    stat_agent = StatisticalAnalysisAgent()
    
    # Получаем датасет
    stat_agent.receive_dataset(test_data)
    
    # Автоматический запуск тестов
    print("=" * 60)
    auto_results = stat_agent.function_calling('auto_run_tests', max_tests=25)
    
    print("\n" + "=" * 60)
    # Получаем значимые результаты в формате строки
    significant_results_str = stat_agent.function_calling('get_significant_results')
    print(significant_results_str)
    
    # Пример отдельных вызовов через function calling
    print("\n" + "=" * 60)
    print("Пример отдельных тестов через function calling:")
    
    # Pearson correlation
    pearson_result = stat_agent.function_calling(
        'pearson_correlation', 
        column1='score_test1', 
        column2='score_test2'
    )
    print(f"\nКорреляция Пирсона: {pearson_result}")
    
    # T-test по группам
    ttest_result = stat_agent.function_calling(
        't_test_independent',
        column1='income',
        group_column='gender'
    )
    print(f"\nT-тест по группам: {ttest_result}")
    
    # Chi-square test
    chi2_result = stat_agent.function_calling(
        'chi_square_test',
        column1='gender',
        column2='education'
    )
    print(f"\nТест хи-квадрат: {chi2_result}")

if __name__ == "__main__":
    main()
