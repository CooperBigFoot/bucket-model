import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd


def plot_water_balance(results: pd.DataFrame, title: str = '', output_destination: str = '', paletette: list = ['#004E64', '007A9A', '00A5CF', '9FFFCB', '25A18E'], start: str = '1986', end: str = '2000') -> None:
    """This function plots the water balance of the model.
    
    Parameters:
    - results (pd.DataFrame): The results from the model run
    - title (str): The title of the plot, if empty, no title will be shown
    - output_destination (str): The path to the output file, if empty, the plot will not be saved
    - paletette (list): The color paletette to use for the plot, default is ['#004E64', '007A9A', '00A5CF', '9FFFCB', '25A18E']
    - start (str): The start year of the plot, default is '1986'
    - end (str): The end year of the plot, default is '2000'
    """
    pass
