import pandas as pd

def analyze_df(df: pd.DataFrame) -> dict:
    summary = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "missing_by_col": df.isna().sum().sort_values(ascending=False).head(20).to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numeric_desc": df.select_dtypes(include=["number"]).describe().to_dict(),
        "top_columns": list(df.columns[:25]),
    }
    return summary
