import pandas as pd
import numpy as np
from scipy import stats
import json

statistical_functions = [
    {
        "name": "pearson_correlation",                #   ПИРСОН КОРРЕЛЯЦИЯ
        "description": "Calculate Pearson correlation between two numeric columns",
        "parameters": {
            "type": "object",
            "properties": {
                "column1": {"type": "string", "description": "First numeric column name"},
                "column2": {"type": "string", "description": "Second numeric column name"}
            },
            "required": ["column1", "column2"]
        }
    },
    {
        "name": "chi_square_test",                        # ХИ КВАДРАТ
        "description": "Perform Chi-square test of independence between two categorical columns",
        "parameters": {
            "type": "object",
            "properties": {
                "column1": {"type": "string", "description": "First categorical column name"},
                "column2": {"type": "string", "description": "Second categorical column name"}
            },
            "required": ["column1", "column2"]
        }
    },
    {
        "name": "t_test_independent",                       # T-TEST
        "description": "Perform independent t-test between two groups",
        "parameters": {
            "type": "object",
            "properties": {
                "numeric_column": {"type": "string", "description": "Numeric column to test"},
                "group_column": {"type": "string", "description": "Categorical column defining groups"}
            },
            "required": ["numeric_column", "group_column"]
        }
    },
    {
        "name": "anova_test",                            #  МНОГОФАКТОРНЫЙ ДИСПЕРСИОННЫЙ АНАЛИЗ
        "description": "Perform ANOVA test for multiple groups",
        "parameters": {
            "type": "object",
            "properties": {
                "numeric_column": {"type": "string", "description": "Numeric column to test"},
                "group_column": {"type": "string", "description": "Categorical column defining groups"}
            },
            "required": ["numeric_column", "group_column"]
        }
    },
]
