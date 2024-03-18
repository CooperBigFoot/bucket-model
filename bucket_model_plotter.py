import seaborn as sns                   # For styling the plots
import matplotlib.pyplot as plt         # For plotting

from scipy.stats import gaussian_kde    # For the density plot
import numpy as np                      # For the density plot

import pandas as pd                     # For the data handling


def plot_water_balance(results: pd.DataFrame, title: str = '', output_destination: str = '', color_palette: list = ['#004E64', '#007A9A', '#00A5CF', '#9FFFCB', '#25A18E'], start: str = '1986', end: str = '2000', figsize: tuple[int, int] = (10, 6), fontsize: int = 12) -> None:
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
    plot_bar_layer(ax, years - BAR_WIDTH / 2, yearly_totals['Rain'], 'Rain', color_palette[0])
    plot_bar_layer(ax, years - BAR_WIDTH / 2, yearly_totals['Snow'], 'Snow', color_palette[1], bottom_layer_heights=yearly_totals['Rain'])
    plot_bar_layer(ax, years + BAR_WIDTH / 2, yearly_totals['Q_s'], 'Q$_{surface}$', color_palette[2])
    plot_bar_layer(ax, years + BAR_WIDTH / 2, yearly_totals['Q_gw'], 'Q$_{gw}$', color_palette[3], bottom_layer_heights=yearly_totals['Q_s'])
    plot_bar_layer(ax, years + BAR_WIDTH / 2, yearly_totals['ET'], 'ET', color_palette[4], bottom_layer_heights=yearly_totals['Q_s'] + yearly_totals['Q_gw'])

    ax.tick_params(which='both', length=10, width=2, labelsize=FONTSIZE)
    ax.set_ylabel('Water depth [mm]', fontsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE, ncol=3, loc='best')
    plt.tight_layout()
    sns.despine()

    if title:
        plt.title(title)

    # Save the plot if an output destination is provided
    if output_destination:
        fig.savefig(output_destination, dpi=300, bbox_inches='tight')


# TODO: Implement the plot_Q_Q function
def plot_Q_Q(results: pd.DataFrame, validation: pd.DataFrame, title: str = '', output_destination: str = '', color: str = '#007A9A', figsize: tuple[int, int] = (6, 6), fontsize: int = 12, line: bool = True, kde: bool = True, cmap: str = 'rainbow') -> None:
    """This function plots the observed vs simulated total runoff (Q) values.
    
    Parameters:
    - results (pd.DataFrame): The results from the model run
    - validation (pd.DataFrame): The validation data. Should the following column: 'Q' for the observed runoff
    - title (str): The title of the plot, if empty, no title will be shown
    - output_destination (str): The path to the output file, if empty, the plot will not be saved
    - color (str): The color of the plot, default is '#007A9A' (a nice blue color)
    - figsize (tuple): The size of the figure, default is (10, 6)
    - fontsize (int): The fontsize of the plot, default is 12
    - line (bool): If True, a 1:1 line will be plotted, default is True
    - kde (bool): If True, a kernel density estimate will be plotted, default is True. Basically colors the points based on the number of points in that area.
      For morre info see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html 
    - cmap (str): The colormap to use for the kde, default is 'rainbow'
    """

    # Some style settings, these is what I like, but feel free to change
    FONTSIZE = fontsize
    sns.set_context('paper')

    # Prepare the data
    results_filtered = results.copy()
    results_filtered['Total_Runoff'] = results_filtered['Q_s'] + results_filtered['Q_gw']

    fig, ax = plt.subplots(figsize=figsize)

    if kde: # If you choose to use the kde, the points will be colored based on the number of points in that area
        xy = np.vstack([results_filtered['Total_Runoff'], validation['Q']])
        z = gaussian_kde(xy)(xy)
        sns.scatterplot(x=results_filtered['Total_Runoff'], y=validation['Q'], ax=ax, c=z, s=30, cmap=cmap, edgecolor='none')

    else: # If you choose not to use the kde, the points will be colored based on the color parameter
        sns.scatterplot(x=results_filtered['Total_Runoff'], y=validation['Q'], ax=ax, color=color, s=30, edgecolor='none')

    if line:
        min_value = min(results_filtered['Total_Runoff'].min(), validation['Q'].min())
        max_value = max(results_filtered['Total_Runoff'].max(), validation['Q'].max())

        ax.plot([min_value, max_value], [min_value, max_value], color='black', linestyle='--')

    # Some more style settings. I recommend keeping these
    ax.set_xlabel('Simulated total runoff [mm/d]', fontsize=FONTSIZE)
    ax.set_ylabel('Observed total runoff [mm/d]', fontsize=FONTSIZE)
    ax.tick_params(which='both', length=10, width=2, labelsize=FONTSIZE)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    sns.despine()


    if title:
        plt.title(title)

    # Save the plot if an output destination is provided
    if output_destination:
        fig.savefig(output_destination, dpi=300, bbox_inches='tight')

# TODO: implement the plot_ECDF function
def plot_ECDF(results: pd.DataFrame, validation: pd.DataFrame, title: str = '', output_destination: str = '', color_palette: list[str, str] = ['#007A9A', '#9FFFCB'], figsize: tuple[int, int] = (6, 6), fontsize: int = 12) -> None:
    """This function plots the empirical cumulative distribution function (ECDF) of the observed and simulated total runoff (Q) values.
    
    Parameters:
    - results (pd.DataFrame): The results from the model run
    - validation (pd.DataFrame): The validation data. Should the following column: 'Q' for the observed runoff
    - title (str): The title of the plot, if empty, no title will be shown
    - output_destination (str): The path to the output file, if empty, the plot will not be saved
    - color_palette (list): The color palette to use for the plot, default is ['#007A9A', '#25A18E']
    - figsize (tuple): The size of the figure, default is (6, 6)
    - fontsize (int): The fontsize of the plot, default is 12
    """

    # Some style settings, these is what I like, but feel free to change
    FONTSIZE = fontsize
    sns.set_context('paper')

    # Prepare the data
    results_filtered = results.copy()
    results_filtered['Total_Runoff'] = results_filtered['Q_s'] + results_filtered['Q_gw']

    fig, ax = plt.subplots(figsize=figsize)

    # Plot the ECDF of the observed and simulated total runoff
    sns.ecdfplot(data=results_filtered['Total_Runoff'], ax=ax, color=color_palette[0], label='Simulated total runoff')
    sns.ecdfplot(data=validation['Q'], ax=ax, color=color_palette[1], label='Observed total runoff')

    # Some more style settings. I recommend keeping these
    ax.set_xlabel('Total runoff [mm/d]', fontsize=FONTSIZE)
    ax.set_ylabel('F cumulative', fontsize=FONTSIZE)
    ax.tick_params(which='both', length=10, width=2, labelsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE, loc='best')
    plt.tight_layout()
    sns.despine()
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    if title:
        plt.title(title)

    # Save the plot if an output destination is provided
    if output_destination:
        fig.savefig(output_destination, dpi=300, bbox_inches='tight')

# TODO: implement the plot_boxplot function
        

# TODO: implement the plot_timeseries function



