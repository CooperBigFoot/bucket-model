import pandas as pd 
import numpy as np


def mae(simulated_Q: pd.DataFrame, observed_Q: pd.Series) -> float:
    """Calculate the Mean Absolute Error (MAE) between observed and simulated Q values.

    Parameters:
    observed_Q (np.ndarray): Array of observed Q values.
    simulated_Q (np.ndarray): Array of simulated Q values.

    Returns:
    float: The MAE value.
    """
    absolute_errors = np.abs(observed_Q - simulated_Q)
    mae_value = np.mean(absolute_errors)

    return mae_value

def rmse(simulated_Q: pd.DataFrame, observed_Q: pd.Series) -> float:
    """Calculate the Root Mean Squared Error (RMSE) between observed and simulated Q values.

    Parameters:
    observed_Q (np.ndarray): Array of observed Q values.
    simulated_Q (np.ndarray): Array of simulated Q values.

    Returns:
    float: The RMSE value.
    """
    squared_errors = (observed_Q - simulated_Q) ** 2
    mse_value = np.mean(squared_errors)
    rmse_value = np.sqrt(mse_value)

    return rmse_value

def nse(simulated_Q: pd.DataFrame, observed_Q: pd.Series) -> float:
    """Calculate the Nash-Sutcliffe Efficiency (NSE) between observed and simulated Q values.

    Parameters:
    observed_Q (np.ndarray): Array of observed Q values.
    simulated_Q (np.ndarray): Array of simulated Q values.

    Returns:
    float: The NSE value.
    """
    numerator = np.sum((observed_Q - simulated_Q) ** 2)
    denominator = np.sum((observed_Q - np.mean(observed_Q)) ** 2)
    nse_value = 1 - (numerator / denominator)

    return nse_value 

def log_nse(simulated_Q: pd.DataFrame, observed_Q: pd.Series) -> float:
    """Calculate the Log Nash-Sutcliffe Efficiency (Log-NSE) between observed and simulated Q values.

    Parameters:
    observed_Q (np.ndarray): Array of observed Q values.
    simulated_Q (np.ndarray): Array of simulated Q values.

    Returns:
    float: The Log-NSE value.
    """

    log_observed_Q = np.log(observed_Q + 1) # Add 1 to avoid log(0)
    log_simulated_Q = np.log(simulated_Q + 1)

    return nse(log_simulated_Q, log_observed_Q) 

def kge(simulated_Q: pd.DataFrame, observed_Q: pd.Series) -> float:
    """Calculate the Kling-Gupta Efficiency (KGE) between observed and simulated Q values.

    Parameters:
    observed_Q (np.ndarray): Array of observed Q values.
    simulated_Q (np.ndarray): Array of simulated Q values.

    Returns:
    float: The KGE value.

    Source: https://en.wikipedia.org/wiki/Kling%E2%80%93Gupta_efficiency
    """
    r = np.corrcoef(observed_Q, simulated_Q)[0, 1]
    alpha = np.std(simulated_Q) / np.std(observed_Q)
    beta = np.mean(simulated_Q) / np.mean(observed_Q)
    kge_value = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    return kge_value 

def pbias(simulated_Q: pd.DataFrame, observed_Q: pd.Series) -> float:
    """Calculate the Percent Bias (PBIAS) between observed and simulated Q values.

    Parameters:
    observed_Q (np.ndarray): Array of observed Q values.
    simulated_Q (np.ndarray): Array of simulated Q values.

    Returns:
    float: The PBIAS value.
    """
    pbias_value = 100 * np.sum(observed_Q - simulated_Q) / np.sum(observed_Q)

    return pbias_value

