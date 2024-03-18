from shutil import which
from tkinter import font
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd


def plot_water_balance(results: pd.DataFrame, title: str = '', output_destination: str = '', palette: list = ['#004E64', '#007A9A', '#00A5CF', '#9FFFCB', '#25A18E'], start: str = '1986', end: str = '2000', figsize: tuple[int, int] = (10, 6), fontsize: int = 12) -> None:
    """This function plots the water balance of the model.
    
    Parameters:
    - results (pd.DataFrame): The results from the model run
    - title (str): The title of the plot, if empty, no title will be shown
    - output_destination (str): The path to the output file, if empty, the plot will not be saved
    - paletette (list): The color paletette to use for the plot, default is ['#004E64', '007A9A', '00A5CF', '9FFFCB', '25A18E']
    - start (str): The start year of the plot, default is '1986'
    - end (str): The end year of the plot, default is '2000'
    - figsize (tuple): The size of the figure, default is (10, 6)
    - fontsize (int): The fontsize of the plot, default is 12
    """

    # Some style settings, these is what I like, but feel free to change
    BAR_WIDTH = .3535
    FONTSIZE = fontsize
    sns.set_context('paper')
    sns.set_style('white')

    # Function to plot a single bar chart layer
    def plot_bar_layer(ax: plt.Axes, positions: int, heights: pd.DataFrame, label: str, color: str, bottom_layer_heights: pd.DataFrame = None) -> None:
        """Helper function to plot a single layer of a bar chart.
        
        Parameters:
        - ax (plt.Axes): The ax to plot on
        - positions (int): The x-positions of the bars
        - heights (pd.DataFrame): The heights of the bars: basically the values to plot
        - label (str): The label of the layer
        - color (str): The color of the layer
        - bottom_layer_heights (pd.DataFrame): The heights of the bottom layer, default is None. Basically the values of the layer below the current layer
        """
        ax.bar(positions, heights, width=BAR_WIDTH, label=label, color=color, bottom=bottom_layer_heights)

    # Prepare the data
    results_filtered = results.copy()
    results_filtered['Year'] = results_filtered.index.year
    results_filtered = results_filtered[start:end]
    yearly_totals = results_filtered.groupby('Year').sum()

    years = yearly_totals.index

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each component of the water balance
    plot_bar_layer(ax, years - BAR_WIDTH / 2, yearly_totals['Rain'], 'Rain', palette[0])
    plot_bar_layer(ax, years - BAR_WIDTH / 2, yearly_totals['Snow'], 'Snow', palette[1], bottom_layer_heights=yearly_totals['Rain'])
    plot_bar_layer(ax, years + BAR_WIDTH / 2, yearly_totals['Q_s'], 'Q$_{surface}$', palette[2])
    plot_bar_layer(ax, years + BAR_WIDTH / 2, yearly_totals['Q_gw'], 'Q$_{gw}$', palette[3], bottom_layer_heights=yearly_totals['Q_s'])
    plot_bar_layer(ax, years + BAR_WIDTH / 2, yearly_totals['ET'], 'ET', palette[4], bottom_layer_heights=yearly_totals['Q_s'] + yearly_totals['Q_gw'])

    ax.tick_params(which='both', length=10, width=2, labelsize=FONTSIZE)
    ax.set_ylabel('Water depth [mm]', fontsize=FONTSIZE)

    ax.legend(fontsize=FONTSIZE, ncol=3, loc='best')
    if title:
        plt.title(title)

    plt.tight_layout()
    sns.despine()

    plt.show()

    # Save the plot if an output destination is provided
    if output_destination:
        fig.savefig(output_destination, dpi=300, bbox_inches='tight')


# TODO: Implement the plot_Q_Q function
def plot_Q_Q(results: pd.DataFrame, title: str = '', output_destination: str = '', color: str = 'blue', fancy: bool = False) -> None:
    """This function plots the observed vs simulated total runoff (Q) values.
    
    Parameters:
    - results (pd.DataFrame): The results from the model run
    - title (str): The title of the plot, if empty, no title will be shown
    - output_destination (str): The path to the output file, if empty, the plot will not be saved
    - color (str): The color of the plot, default is 'blue'
    - fancy (bool): If True, the plot gets a fancy look, default is False. Inspiration: https://seaborn.pydata.org/examples/layered_bivariate_plot.html
    """
    pass
