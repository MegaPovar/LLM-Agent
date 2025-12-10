import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Tuple
import warnings
import json
warnings.filterwarnings('ignore')

class StatAgent:
    """
    Агент статистического анализа для работы в агентной системе.
    Принимает промпт с описанием датасета и определяет, какие тесты выполнить.
    """
    
    def __init__(self):
        self.dataset = None
        self.dataset_description = ""
        self.test_results = []
        self.significant_results = []
        self.planned_tests = []
        self.MAX_TESTS = 30  # МАКСИМАЛЬНОЕ количество тестов
    
    def receive_prompt(self, prompt: str, data: pd.DataFrame):
        """
        Принять промпт с описанием датасета и сам датасет
        """
        self.dataset_description = prompt
        self.dataset = data
        
        print(f"Агент получил промпт: {prompt[:100]}...")
        print(f"Датасет: {data.shape[0]} строк, {data.shape[1]} колонок")
        
        # Анализируем датасет и планируем тесты
        self._plan_tests_based_on_description()
        
        return {
            "status": "success",
            "message": f"Промпт обработан. Запланировано {len(self.planned_tests)} тестов.",
            "planned_tests": len(self.planned_tests)
        }
    
    def _analyze_dataset_structure(self):
        """Анализ структуры датасета"""
        column_info = {}
        for col in self.dataset.columns:
            dtype = str(self.dataset[col].dtype)
            unique_count = self.dataset[col].nunique()
            non_null_count = self.dataset[col].count()
            
            is_numeric = dtype.startswith(('int', 'float'))
            
            column_info[col] = {
                'dtype': dtype,
                'unique_count': unique_count,
                'non_null_count': non_null_count,
                'is_numeric': is_numeric,
                'is_binary': is_numeric and unique_count == 2,
                'is_categorical': (not is_numeric and unique_count <= 20) or 
                                 (is_numeric and 2 < unique_count <= 10),
                'is_ordinal': is_numeric and 2 < unique_count <= 7,
                'sample_values': list(self.dataset[col].dropna().unique()[:3]) if unique_count > 1 else []
            }
        
        self.column_info = column_info
        return column_info
    
    def _plan_tests_based_on_description(self):
        """
        На основе описания в промпте определить, какие тесты стоит выполнить
        """
        # Анализируем структуру данных
        self._analyze_dataset_structure()
        
        # Извлекаем ключевые слова из описания
        description_lower = self.dataset_description.lower()
        
        # Определяем типы переменных из описания
        numeric_cols = []
        categorical_cols = []
        binary_cols = []
        target_columns = []
        
        # Автоматическое определение колонок по их именам и типам
        for col, info in self.column_info.items():
            col_lower = col.lower()
            
            if info['is_numeric']:
                if info['is_binary']:
                    binary_cols.append(col)
                else:
                    numeric_cols.append(col)
                    
                    # Предполагаем, что некоторые числовые колонки могут быть целевыми
                    if any(word in col_lower for word in ['score', 'result', 'rating', 'price', 'cost', 
                                                         'amount', 'revenue', 'profit', 'loss', 'target',
                                                         'outcome', 'performance', 'efficiency']):
                        target_columns.append(col)
            elif info['is_categorical'] or info['unique_count'] <= 10:
                categorical_cols.append(col)
                
                # Предполагаем, что некоторые категориальные колонки могут быть группирующими
                if any(word in col_lower for word in ['group', 'type', 'category', 'class', 'segment',
                                                     'gender', 'sex', 'age_group', 'region', 'country',
                                                     'department', 'team', 'level', 'status']):
                    target_columns.append(col)
        
        # Если целевые колонки не найдены автоматически, выбираем наиболее вероятные
        if not target_columns:
            # Берем первую числовую колонку как потенциальную целевую
            if numeric_cols:
                target_columns.append(numeric_cols[0])
            # И первую категориальную как потенциальную группирующую
            if categorical_cols:
                target_columns.append(categorical_cols[0])
        
        # Ограничиваем количество колонок для анализа
        max_cols_per_type = 8
        numeric_cols = numeric_cols[:max_cols_per_type]
        categorical_cols = categorical_cols[:max_cols_per_type]
        binary_cols = binary_cols[:max_cols_per_type]
        
        print(f"Определены колонки: {len(numeric_cols)} числовых, {len(categorical_cols)} категориальных")
        
        # Планируем тесты на основе анализа
        self.planned_tests = []
        
        # 1. Корреляции между числовыми переменными (до 10 тестов)
        if len(numeric_cols) >= 2:
            correlation_pairs = self._generate_meaningful_pairs(numeric_cols, max_pairs=10)
            for col1, col2 in correlation_pairs:
                # Планируем оба типа корреляций для важных пар
                self.planned_tests.append({
                    'test': 'pearson_correlation',
                    'params': {'column1': col1, 'column2': col2},
                    'reason': f'Корреляция между {col1} и {col2}'
                })
                
                # Для разнообразия, каждую вторую пару проверяем также Спирменом
                if len(self.planned_tests) % 3 == 0:
                    self.planned_tests.append({
                        'test': 'spearman_correlation',
                        'params': {'column1': col1, 'column2': col2},
                        'reason': f'Ранговая корреляция между {col1} и {col2}'
                    })
        
        # 2. T-тесты: числовые переменные по категориальным группам (до 8 тестов)
        if numeric_cols and categorical_cols:
            ttest_count = 0
            for num_col in numeric_cols[:4]:  # Ограничиваем числовые колонки
                if ttest_count >= 8:
                    break
                for cat_col in categorical_cols[:3]:  # Ограничиваем категориальные
                    if ttest_count >= 8:
                        break
                    
                    # Проверяем, достаточно ли данных в группах
                    group_sizes = self._get_group_sizes(num_col, cat_col)
                    if group_sizes['min_group_size'] >= 5 and len(group_sizes['groups']) >= 2:
                        self.planned_tests.append({
                            'test': 't_test_independent',
                            'params': {'column1': num_col, 'group_column': cat_col},
                            'reason': f'Сравнение {num_col} по группам {cat_col}'
                        })
                        ttest_count += 1
        
        # 3. Хи-квадрат тесты между категориальными переменными (до 6 тестов)
        if len(categorical_cols) >= 2:
            chi2_count = 0
            for i in range(min(4, len(categorical_cols))):
                for j in range(i+1, min(5, len(categorical_cols))):
                    if chi2_count >= 6:
                        break
                    
                    # Проверяем, что таблица сопряженности не будет слишком разреженной
                    if self._check_chi2_feasibility(categorical_cols[i], categorical_cols[j]):
                        self.planned_tests.append({
                            'test': 'chi_square_test',
                            'params': {'column1': categorical_cols[i], 'column2': categorical_cols[j]},
                            'reason': f'Связь между {categorical_cols[i]} и {categorical_cols[j]}'
                        })
                        chi2_count += 1
        
        # 4. Парные t-тесты для связанных измерений (до 4 тестов)
        # Ищем пары колонок, которые могут быть связанными измерениями
        paired_cols = self._find_potential_paired_columns(numeric_cols)
        for col1, col2 in paired_cols[:4]:
            self.planned_tests.append({
                'test': 't_test_paired',
                'params': {'column1': col1, 'column2': col2},
                'reason': f'Сравнение связанных измерений {col1} и {col2}'
            })
        
        # 5. Дополняем до MAX_TESTS, если нужно
        self._fill_up_tests_to_max()
        
        # Обрезаем до MAX_TESTS, если переполнили
        self.planned_tests = self.planned_tests[:self.MAX_TESTS]
        
        print(f"Запланировано тестов: {len(self.planned_tests)}")
        
        # Выводим план тестов
        print("\nПлан статистических тестов:")
        for i, test in enumerate(self.planned_tests, 1):
            print(f"{i}. {test['test']}: {test['reason']}")
    
    def _generate_meaningful_pairs(self, columns: List[str], max_pairs: int = 10) -> List[Tuple]:
        """Генерация осмысленных пар для анализа"""
        pairs = []
        
        # Приоритетные пары на основе названий колонок
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns[i+1:], i+1):
                if len(pairs) >= max_pairs:
                    break
                
                # Проверяем осмысленность пары
                if self._is_meaningful_pair(col1, col2):
                    pairs.append((col1, col2))
        
        return pairs[:max_pairs]
    
    def _is_meaningful_pair(self, col1: str, col2: str) -> bool:
        """Проверка, имеет ли смысл сравнивать две колонки"""
        try:
            # Минимальное количество данных
            non_null_data = self.dataset[[col1, col2]].dropna()
            if len(non_null_data) < 10:
                return False
            
            # Не постоянные значения
            if self.dataset[col1].nunique() <= 1 or self.dataset[col2].nunique() <= 1:
                return False
            
            # Проверка на возможную зависимость по названиям
            col1_lower = col1.lower()
            col2_lower = col2.lower()
            
            # Пары, которые вероятно связаны
            meaningful_patterns = [
                (['age', 'income'], ['age', 'salary']),
                (['test', 'result'], ['exam', 'score']),
                (['height', 'weight'], ['price', 'quality']),
                (['time', 'score'], ['duration', 'performance']),
            ]
            
            for pattern in meaningful_patterns:
                for word1 in pattern[0]:
                    for word2 in pattern[1]:
                        if word1 in col1_lower and word2 in col2_lower:
                            return True
                        if word2 in col1_lower and word1 in col2_lower:
                            return True
            
            return True
        except:
            return False
    
    def _get_group_sizes(self, numeric_col: str, group_col: str) -> dict:
        """Получить размеры групп для t-теста"""
        try:
            groups = self.dataset.groupby(group_col)[numeric_col].apply(
                lambda x: x.dropna().count()
            ).to_dict()
            return {
                'groups': groups,
                'min_group_size': min(groups.values()) if groups else 0,
                'group_count': len(groups)
            }
        except:
            return {'groups': {}, 'min_group_size': 0, 'group_count': 0}
    
    def _check_chi2_feasibility(self, col1: str, col2: str) -> bool:
        """Проверка возможности выполнения хи-квадрат теста"""
        try:
            contingency = pd.crosstab(self.dataset[col1], self.dataset[col2])
            
            # Проверяем размер таблицы
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                return False
            
            # Проверяем, что не более 20% ячеек имеют ожидаемую частоту < 5
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            low_freq_cells = (expected < 5).sum()
            total_cells = expected.size
            
            if low_freq_cells / total_cells > 0.2:
                return False
            
            return True
        except:
            return False
    
    def _find_potential_paired_columns(self, numeric_cols: List[str]) -> List[Tuple]:
        """Найти пары колонок для парных тестов"""
        pairs = []
        
        # Ищем колонки с похожими названиями
        for i, col1 in enumerate(numeric_cols):
            col1_lower = col1.lower()
            
            for j, col2 in enumerate(numeric_cols[i+1:], i+1):
                col2_lower = col2.lower()
                
                # Пары типа pre/post, before/after, test1/test2
                paired_keywords = [
                    (['pre', 'before', 'начал', 'первый'], ['post', 'after', 'после', 'второй']),
                    (['test1', 'exam1', 'первый'], ['test2', 'exam2', 'второй']),
                    (['time1', 'moment1'], ['time2', 'moment2']),
                    (['score1', 'result1'], ['score2', 'result2']),
                ]
                
                for pre_words, post_words in paired_keywords:
                    pre_match = any(word in col1_lower for word in pre_words)
                    post_match = any(word in col2_lower for word in post_words)
                    
                    if pre_match and post_match:
                        pairs.append((col1, col2))
                        break
                    
                    # Проверяем в обратном порядке
                    pre_match = any(word in col2_lower for word in pre_words)
                    post_match = any(word in col1_lower for word in post_words)
                    
                    if pre_match and post_match:
                        pairs.append((col2, col1))
                        break
        
        return pairs
    
    def _fill_up_tests_to_max(self):
        """Дополнить список тестов до MAX_TESTS, если нужно"""
        if len(self.planned_tests) >= self.MAX_TESTS:
            return
        
        # Анализируем, каких тестов не хватает
        test_types = [t['test'] for t in self.planned_tests]
        
        # Если мало корреляций, добавляем
        correlation_tests = [t for t in self.planned_tests if 'correlation' in t['test']]
        if len(correlation_tests) < 8:
            self._add_more_correlation_tests(8 - len(correlation_tests))
        
        # Если мало t-тестов, добавляем
        t_tests = [t for t in self.planned_tests if 't_test' in t['test']]
        if len(t_tests) < 8:
            self._add_more_ttests(8 - len(t_tests))
        
        # Обрезаем снова до MAX_TESTS
        self.planned_tests = self.planned_tests[:self.MAX_TESTS]
    
    def _add_more_correlation_tests(self, count: int):
        """Добавить дополнительные корреляционные тесты"""
        numeric_cols = [col for col, info in self.column_info.items() 
                       if info['is_numeric'] and not info['is_binary']]
        
        added = 0
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                if added >= count:
                    return
                
                col1, col2 = numeric_cols[i], numeric_cols[j]
                
                # Проверяем, что эта пара еще не тестировалась
                pair_exists = any(
                    t['params'].get('column1') == col1 and t['params'].get('column2') == col2 
                    or t['params'].get('column1') == col2 and t['params'].get('column2') == col1
                    for t in self.planned_tests if 'correlation' in t['test']
                )
                
                if not pair_exists and self._is_meaningful_pair(col1, col2):
                    test_type = 'spearman_correlation' if added % 2 == 0 else 'pearson_correlation'
                    self.planned_tests.append({
                        'test': test_type,
                        'params': {'column1': col1, 'column2': col2},
                        'reason': f'Корреляция {col1} и {col2}'
                    })
                    added += 1
    
    def _add_more_ttests(self, count: int):
        """Добавить дополнительные t-тесты"""
        numeric_cols = [col for col, info in self.column_info.items() 
                       if info['is_numeric'] and not info['is_binary']]
        categorical_cols = [col for col, info in self.column_info.items() 
                          if info['is_categorical']]
        
        added = 0
        for num_col in numeric_cols[:6]:
            for cat_col in categorical_cols[:4]:
                if added >= count:
                    return
                
                # Проверяем, что этот тест еще не запланирован
                test_exists = any(
                    t['params'].get('column1') == num_col and t['params'].get('group_column') == cat_col
                    for t in self.planned_tests if 't_test' in t['test']
                )
                
                if not test_exists:
                    group_sizes = self._get_group_sizes(num_col, cat_col)
                    if group_sizes['min_group_size'] >= 5:
                        self.planned_tests.append({
                            'test': 't_test_independent',
                            'params': {'column1': num_col, 'group_column': cat_col},
                            'reason': f'Сравнение {num_col} между группами {cat_col}'
                        })
                        added += 1
    
    def function_calling(self, function_name: str, **kwargs):
        """
        Система function calling для выполнения статистических тестов
        """
        function_map = {
            't_test_independent': self._perform_t_test_independent,
            't_test_paired': self._perform_t_test_paired,
            'chi_square_test': self._perform_chi_square_test,
            'pearson_correlation': self._perform_pearson_correlation,
            'spearman_correlation': self._perform_spearman_correlation,
            'run_planned_tests': self._run_planned_tests,
            'get_significant_results': self._get_significant_results_string
        }
        
        if function_name in function_map:
            result = function_map[function_name](**kwargs)
            
            # Сохраняем результаты тестов
            if function_name not in ['run_planned_tests', 'get_significant_results']:
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
                        'result': result,
                        'description': f"{function_name}: {kwargs}"
                    })
            
            return result
        else:
            return {"error": f"Функция {function_name} не найдена"}
    
    def _perform_t_test_independent(self, column1: str, column2: str = None, group_column: str = None, group1: Any = None, group2: Any = None):
        """Независимый t-тест"""
        try:
            if group_column:
                # Определяем группы, если не указаны
                if group1 is None or group2 is None:
                    groups = self.dataset[group_column].dropna().unique()
                    if len(groups) >= 2:
                        group1, group2 = groups[:2]
                    else:
                        return {"error": "Недостаточно групп для сравнения"}
                
                data1 = self.dataset[self.dataset[group_column] == group1][column1].dropna()
                data2 = self.dataset[self.dataset[group_column] == group2][column1].dropna()
                
                t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
                
                return {
                    "test": "independent_t_test",
                    "column": column1,
                    "group_column": group_column,
                    "groups": [str(group1), str(group2)],
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                    "n1": len(data1),
                    "n2": len(data2),
                    "mean1": float(data1.mean()),
                    "mean2": float(data2.mean()),
                    "std1": float(data1.std()),
                    "std2": float(data2.std())
                }
            else:
                return {"error": "Не указана группирующая переменная"}
                
        except Exception as e:
            return {"error": f"Ошибка t-теста: {str(e)}"}
    
    def _perform_t_test_paired(self, column1: str, column2: str):
        """Парный t-тест"""
        try:
            data1 = self.dataset[column1].dropna()
            data2 = self.dataset[column2].dropna()
            
            min_len = min(len(data1), len(data2))
            data1 = data1.iloc[:min_len]
            data2 = data2.iloc[:min_len]
            
            t_stat, p_value = stats.ttest_rel(data1, data2)
            
            return {
                "test": "paired_t_test",
                "columns": [column1, column2],
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "n_pairs": min_len,
                "mean1": float(data1.mean()),
                "mean2": float(data2.mean()),
                "mean_diff": float((data1 - data2).mean()),
                "std_diff": float((data1 - data2).std())
            }
        except Exception as e:
            return {"error": f"Ошибка парного t-теста: {str(e)}"}
    
    def _perform_chi_square_test(self, column1: str, column2: str):
        """Тест хи-квадрат"""
        try:
            contingency = pd.crosstab(self.dataset[column1], self.dataset[column2])
            
            if contingency.size == 0:
                return {"error": "Пустая таблица сопряженности"}
            
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            
            return {
                "test": "chi_square_test",
                "columns": [column1, column2],
                "chi2_statistic": float(chi2),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "degrees_of_freedom": int(dof),
                "contingency_shape": contingency.shape,
                "total_observations": contingency.sum().sum()
            }
        except Exception as e:
            return {"error": f"Ошибка хи-квадрат теста: {str(e)}"}
    
    def _perform_pearson_correlation(self, column1: str, column2: str):
        """Корреляция Пирсона"""
        try:
            data1 = self.dataset[column1].dropna()
            data2 = self.dataset[column2].dropna()
            
            min_len = min(len(data1), len(data2))
            data1 = data1.iloc[:min_len]
            data2 = data2.iloc[:min_len]
            
            if min_len < 3:
                return {"error": "Недостаточно данных для корреляции"}
            
            corr_coef, p_value = stats.pearsonr(data1, data2)
            
            return {
                "test": "pearson_correlation",
                "columns": [column1, column2],
                "correlation_coefficient": float(corr_coef),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "n": min_len,
                "strength": self._interpret_correlation(corr_coef)
            }
        except Exception as e:
            return {"error": f"Ошибка корреляции Пирсона: {str(e)}"}
    
    def _perform_spearman_correlation(self, column1: str, column2: str):
        """Корреляция Спирмена"""
        try:
            data1 = self.dataset[column1].dropna()
            data2 = self.dataset[column2].dropna()
            
            min_len = min(len(data1), len(data2))
            data1 = data1.iloc[:min_len]
            data2 = data2.iloc[:min_len]
            
            if min_len < 3:
                return {"error": "Недостаточно данных для корреляции"}
            
            corr_coef, p_value = stats.spearmanr(data1, data2)
            
            return {
                "test": "spearman_correlation",
                "columns": [column1, column2],
                "correlation_coefficient": float(corr_coef),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "n": min_len,
                "strength": self._interpret_correlation(corr_coef)
            }
        except Exception as e:
            return {"error": f"Ошибка корреляции Спирмена: {str(e)}"}
    
    def _interpret_correlation(self, r):
        """Интерпретация корреляции"""
        r_abs = abs(r)
        if r_abs >= 0.9:
            return "очень сильная"
        elif r_abs >= 0.7:
            return "сильная"
        elif r_abs >= 0.5:
            return "умеренная"
        elif r_abs >= 0.3:
            return "слабая"
        else:
            return "очень слабая"
    
    def _run_planned_tests(self):
        """Выполнить все запланированные тесты"""
        if not self.planned_tests:
            return {"error": "Нет запланированных тестов"}
        
        print(f"Выполнение {len(self.planned_tests)} запланированных тестов...")
        
        executed_tests = []
        successful_tests = 0
        
        for i, test_info in enumerate(self.planned_tests, 1):
            print(f"Тест {i}/{len(self.planned_tests)}: {test_info['test']} - {test_info['reason']}")
            
            try:
                result = self.function_calling(test_info['test'], **test_info['params'])
                
                if 'error' not in result:
                    successful_tests += 1
                    if result.get('significant', False):
                        print(f"  ✓ Статистически значимый (p={result.get('p_value', 0):.4f})")
                    else:
                        print(f"  × Незначимый (p={result.get('p_value', 1):.4f})")
                else:
                    print(f"  ! Ошибка: {result.get('error', 'unknown')}")
                
                executed_tests.append({
                    'test_number': i,
                    'test_info': test_info,
                    'result': result
                })
                
            except Exception as e:
                print(f"  ! Исключение: {str(e)}")
                executed_tests.append({
                    'test_number': i,
                    'test_info': test_info,
                    'result': {"error": str(e)}
                })
        
        print(f"\nРезультаты: {successful_tests} успешных тестов из {len(self.planned_tests)}")
        print(f"Значимых результатов: {len(self.significant_results)}")
        
        return {
            "total_tests": len(self.planned_tests),
            "successful_tests": successful_tests,
            "significant_tests": len(self.significant_results),
            "executed_tests": executed_tests
        }
    
    def _get_significant_results_string(self):
        """Получить строку со значимыми результатами для другого агента"""
        if not self.significant_results:
            return "STAT_RESULTS: Нет статистически значимых результатов (p < 0.05)"
        
        result_lines = ["STAT_RESULTS: Статистически значимые зависимости (p < 0.05):"]
        
        for i, res in enumerate(self.significant_results, 1):
            test_type = res['test']
            params = res['parameters']
            result = res['result']
            
            if 'correlation' in test_type:
                col1 = params.get('column1', '')
                col2 = params.get('column2', '')
                corr = result.get('correlation_coefficient', 0)
                p_val = result.get('p_value', 1)
                strength = result.get('strength', '')
                
                line = (f"{i}. {test_type}: {col1} ↔ {col2} | "
                       f"r = {corr:.3f} ({strength}), p = {p_val:.4f} ***")
            
            elif 't_test' in test_type:
                if 'group_column' in params:
                    col = params.get('column1', '')
                    group_col = params.get('group_column', '')
                    groups = result.get('groups', ['', ''])
                    t_stat = result.get('t_statistic', 0)
                    p_val = result.get('p_value', 1)
                    
                    line = (f"{i}. {test_type}: {col} по {group_col} ({groups[0]} vs {groups[1]}) | "
                           f"t = {t_stat:.3f}, p = {p_val:.4f} ***")
                else:
                    cols = result.get('columns', ['', ''])
                    t_stat = result.get('t_statistic', 0)
                    p_val = result.get('p_value', 1)
                    
                    line = (f"{i}. {test_type}: {cols[0]} ↔ {cols[1]} | "
                           f"t = {t_stat:.3f}, p = {p_val:.4f} ***")
            
            elif test_type == 'chi_square_test':
                cols = result.get('columns', ['', ''])
                chi2 = result.get('chi2_statistic', 0)
                p_val = result.get('p_value', 1)
                
                line = (f"{i}. chi_square: {cols[0]} ↔ {cols[1]} | "
                       f"χ² = {chi2:.3f}, p = {p_val:.4f} ***")
            
            else:
                line = f"{i}. {test_type}: p = {result.get('p_value', 0):.4f} ***"
            
            result_lines.append(line)
        
        return "\n".join(result_lines)


# Пример использования
def main():
    # Создаем тестовый датасет
    np.random.seed(42)
    n_samples = 150
    
    test_data = pd.DataFrame({
        'age': np.random.normal(35, 10, n_samples),
        'salary': np.random.normal(50000, 15000, n_samples),
        'test_score_pre': np.random.normal(70, 10, n_samples),
        'test_score_post': np.random.normal(75, 9, n_samples) + np.random.randn(n_samples) * 3,
        'performance_rating': np.random.normal(4.0, 1.0, n_samples),
        'hours_worked': np.random.normal(40, 5, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4]),
        'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'department': np.random.choice(['Sales', 'Engineering', 'Marketing', 'HR'], n_samples),
        'satisfaction_score': np.random.randint(1, 6, n_samples),
        'project_completed': np.random.randint(0, 20, n_samples),
        'is_manager': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    })
    
    # Добавляем некоторые зависимости
    test_data.loc[test_data['gender'] == 'Male', 'salary'] += 5000
    test_data['test_score_post'] = test_data['test_score_pre'] * 0.8 + np.random.randn(n_samples) * 8 + 10
    test_data.loc[test_data['education_level'] == 'PhD', 'performance_rating'] += 0.5
    
    # Создаем агента
    agent = StatisticalAnalysisAgent()
    
    # Промпт от другого агента
    prompt = """
    У тебя есть датасет сотрудников компании. 
    Содержит данные о возрасте, зарплате, результатах тестов до и после обучения, 
    оценке производительности, количестве отработанных часов, поле, уровне образования,
    отделе работы, оценке удовлетворенности и количестве завершенных проектов.
    Также есть бинарная переменная is_manager (является ли руководителем).
    
    Проанализируй возможные взаимосвязи:
    1. Влияние пола и образования на зарплату
    2. Корреляция между возрастом, зарплатой и производительностью
    3. Эффективность обучения (сравнение test_score_pre и test_score_post)
    4. Связь между отделом, должностью и удовлетворенностью
    """
    
    # Агент получает промпт и данные
    agent.receive_prompt(prompt, test_data)
    
    # Запускаем запланированные тесты через function calling
    results = agent.function_calling('run_planned_tests')
    
    print("\n" + "="*60)
    # Получаем значимые результаты в формате строки
    significant_results = agent.function_calling('get_significant_results')
    print(significant_results)

if __name__ == "__main__":
    main()
