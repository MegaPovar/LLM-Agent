import pandas as pd
import json
import requests
from io import StringIO

def get_dataframe_info(df):
    description = {
        "shape": df.shape,
        "columns": list(df.columns),
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "sample_data": df.head(5).to_dict('records'),
    }
    return description
