import pandas as pd
import numpy as np

def get_partition_limits(df: pd.DataFrame, column: str, npartitions: int = 3) -> np.ndarray:
    """
    Compute the partition limits for a given numeric column in a DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The name of the column for which to calculate partition limits.
    Returns:
        np.ndarray: An array containing the 1/3 and 2/3 quantiles of the specified column.
    Raises:
        TypeError: If the specified column is not numeric.
    """
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"Column '{column}' must be numeric.")
    quantile_points = [i / npartitions for i in range(1, npartitions)]
    quantiles = df[column].quantile(quantile_points).values
    return quantiles


def split_dataframe_by_quantiles(df: pd.DataFrame, column: str, quantiles: np.ndarray) -> list:
    """
    Split the DataFrame into partitions based on quantile limits for a given column.

    Args:
        df (pd.DataFrame): The DataFrame to split.
        column (str): The column to use for quantile-based splitting.
        quantiles (np.ndarray): The quantile limits to use for splitting.

    Returns:
        list: A list of DataFrames, each corresponding to a partition.
    """
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"Column '{column}' must be numeric.")
    partitions = []
    prev_limit = -np.inf
    for limit in quantiles:
        partitions.append(df[(df[column] > prev_limit) & (df[column] <= limit)])
        prev_limit = limit
    partitions.append(df[df[column] > prev_limit])
    return partitions

def split_dataframe_by_values(df: pd.DataFrame, column: str) -> list:
    """
    Split the DataFrame into partitions based on unique string values of a given column.

    Args:
        df (pd.DataFrame): The DataFrame to split.
        column (str): The column to use for value-based splitting.

    Returns:
        list: A list of DataFrames, each corresponding to a unique value in the column.
    Raises:
        TypeError: If the specified column is not of string type.
    """
    if not pd.api.types.is_string_dtype(df[column]):
        raise TypeError(f"Column '{column}' must be of string type.")
    return [df[df[column] == value] for value in df[column].unique()]