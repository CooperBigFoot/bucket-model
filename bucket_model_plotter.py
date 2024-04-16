import seaborn as sns                   # For styling the plots
import matplotlib.pyplot as plt         # For plotting

from scipy.stats import gaussian_kde    # For the density plot
import numpy as np                      # For the density plot

import pandas as pd                     # For the data handling


def plot_water_balance(results: pd.DataFrame, title: str = '', output_destination: str = '', palette: list = ['#004E64', '#007A9A', '#00A5CF', '#9FFFCB', '#25A18E'], start: str = '1986', end: str = '2000', figsize: tuple[int, int] = (10, 6), fontsize: int = 12) -> None:
    """This function plots the water balance of the model.
    
    Parameters:
    - results (pd.DataFrame): The results from the model run
    - title (str): The title of the plot, if empty, no title will be shown
    - output_destination (str): The path to the output file, if empty, the plot will not be saved
    - palette (list): The color palette to use for the plot, default is ['#004E64', '007A9A', '00A5CF', '9FFFCB', '25A18E']
    - start (str): The start year of the plot, default is '1986'
    - end (str): The end year of the plot, default is '2000'
    - figsize (tuple): The size of the figure, default is (10, 6)
    - fontsize (int): The fontsize of the plot, default is 12
    """

    # Some style settings, this is what I like, but feel free to change it
    BAR_WIDTH = .35
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

    ax.tick_params(which='both', length=10, width=2, labelsize=fontsize)
    ax.set_ylabel('Water depth [mm]', fontsize=fontsize)
    ax.legend(fontsize=fontsize, ncol=3, loc='best')
    plt.tight_layout()
    sns.despine()

    if title:
        plt.title(title)

    # Save the plot if an output destination is provided
    if output_destination:
        fig.savefig(output_destination, dpi=300, bbox_inches='tight')


def plot_Q_Q(results: pd.DataFrame, observed: pd.DataFrame, title: str = '', output_destination: str = '', color: str = '#007A9A', figsize: tuple[int, int] = (6, 6), fontsize: int = 12, line: bool = True, kde: bool = True, cmap: str = 'rainbow') -> None:
    """This function plots the observed vs simulated total runoff (Q) values.
    
    Parameters:
    - results (pd.DataFrame): The results from the model run
    - observed (pd.DataFrame): The observed data. Should contain the following column: 'Q' for the observed runoff
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

    # Some style settings, this is what I like, but feel free to change it
    sns.set_context('paper')

    # Prepare the data
    results_filtered = results.copy()
    results_filtered['Total_Runoff'] = results_filtered['Q_s'] + results_filtered['Q_gw']

    fig, ax = plt.subplots(figsize=figsize)

    if kde: # If you choose to use the kde, the points will be colored based on the number of points in that area
        xy = np.vstack([results_filtered['Total_Runoff'], observed['Q']])
        z = gaussian_kde(xy)(xy)
        sns.scatterplot(x=results_filtered['Total_Runoff'], y=observed['Q'], ax=ax, c=z, s=30, cmap=cmap, edgecolor='none')

    else: # If you choose not to use the kde, the points will be colored based on the color parameter
        sns.scatterplot(x=results_filtered['Total_Runoff'], y=observed['Q'], ax=ax, color=color, s=30, edgecolor='none')

    if line:
        min_value = min(results_filtered['Total_Runoff'].min(), observed['Q'].min())
        max_value = max(results_filtered['Total_Runoff'].max(), observed['Q'].max())

        ax.plot([min_value, max_value], [min_value, max_value], color='black', linestyle='--')

    # Some more style settings. I recommend keeping this
    ax.set_xlabel('Simulated total runoff [mm/d]', fontsize=fontsize)
    ax.set_ylabel('Observed total runoff [mm/d]', fontsize=fontsize)
    ax.tick_params(which='both', length=10, width=2, labelsize=fontsize)
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

def plot_ECDF(results: pd.DataFrame, observed: pd.DataFrame, title: str = '', output_destination: str = '', palette: list[str, str] = ['#007A9A', '#9FFFCB'], figsize: tuple[int, int] = (6, 6), fontsize: int = 12) -> None:
    """This function plots the empirical cumulative distribution function (ECDF) of the observed and simulated total runoff (Q) values.
    
    Parameters:
    - results (pd.DataFrame): The results from the model run
    - observed (pd.DataFrame): The observed data. Should contain the following column: 'Q' for the observed runoff
    - title (str): The title of the plot, if empty, no title will be shown
    - output_destination (str): The path to the output file, if empty, the plot will not be saved
    - palette (list): The color palette to use for the plot, default is ['#007A9A', '#25A18E']
    - figsize (tuple): The size of the figure, default is (6, 6)
    - fontsize (int): The fontsize of the plot, default is 12
    """

    # Some style settings, this is what I like, but feel free to change it
    sns.set_context('paper')

    # Prepare the data
    results_filtered = results.copy()
    results_filtered['Total_Runoff'] = results_filtered['Q_s'] + results_filtered['Q_gw']

    fig, ax = plt.subplots(figsize=figsize)

    # Plot the ECDF of the observed and simulated total runoff
    sns.ecdfplot(data=results_filtered['Total_Runoff'], ax=ax, color=palette[0], label='Simulated total runoff')
    sns.ecdfplot(data=observed['Q'], ax=ax, color=palette[1], label='Observed total runoff')

    # Some more style settings. I recommend keeping this
    ax.set_xlabel('Total runoff [mm/d]', fontsize=fontsize)
    ax.set_ylabel('F cumulative', fontsize=fontsize)
    ax.tick_params(which='both', length=10, width=2, labelsize=fontsize)
    ax.legend(fontsize=fontsize, loc='best')
    plt.tight_layout()
    sns.despine()
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    if title:
        plt.title(title)

    # Save the plot if an output destination is provided
    if output_destination:
        fig.savefig(output_destination, dpi=300, bbox_inches='tight')

def plot_KDE(results: pd.DataFrame, observed: pd.DataFrame, title: str = '', output_destination: str = '', palette: list[str, str] = ['#007A9A', '#25A18E'], figsize: tuple[int, int] = (6, 6), fontsize: int = 12, fill: bool = True) -> None:
    """This function plots the kernel density estimate (KDE) of the observed and simulated total runoff (Q) values.
    
    Parameters:
    - results (pd.DataFrame): The results from the model run
    - observed (pd.DataFrame): The observed data. Should contain the following column: 'Q' for the observed runoff
    - title (str): The title of the plot, if empty, no title will be shown
    - output_destination (str): The path to the output file, if empty, the plot will not be saved
    - palette (list): The color palette to use for the plot, default is ['#007A9A', '#25A18E']
    - figsize (tuple): The size of the figure, default is (6, 6)
    - fontsize (int): The fontsize of the plot, default is 12
    - fill (bool): If True, the KDE will be filled, default is True
    """

    # Some style settings, this is what I like, but feel free to change it
    sns.set_context('paper')

    # Prepare the data
    results_filtered = results.copy()
    results_filtered['Total_Runoff'] = results_filtered['Q_s'] + results_filtered['Q_gw']

    fig, ax = plt.subplots(figsize=figsize)

    # Plot the KDE of the observed and simulated total runoff
    sns.kdeplot(data=results_filtered['Total_Runoff'], ax=ax, color=palette[0], label='Simulated total runoff', fill=fill)
    sns.kdeplot(data=observed['Q'], ax=ax, color=palette[1], label='Observed total runoff', fill=fill)

    # Some more style settings. I recommend keeping this
    ax.set_xlabel('Total runoff [mm/d]', fontsize=fontsize)
    ax.set_ylabel('Density', fontsize=fontsize)
    ax.tick_params(which='both', length=10, width=2, labelsize=fontsize)
    ax.legend(fontsize=fontsize, loc='best')
    plt.tight_layout()
    sns.despine()
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    if title:
        plt.title(title)

    # Save the plot if an output destination is provided
    if output_destination:
        fig.savefig(output_destination, dpi=300, bbox_inches='tight')


def plot_monthly_boxplot(results: pd.DataFrame, title: str = '', output_destination: str = '', figsize: tuple[int, int] = (12, 12), fontsize: int = 12, palette: list[str, str, str, str] = ['#004E64', '#007A9A', '#00A5CF', '#9FFFCB']) -> None:
    """This function plots the monthly boxplot of the simulated environmental variables: 
    - Monthly Precipitation 
    - Actual Monthly Evapotranspiration
    - Monthly Snowmelt
    - Monthly simulated Total Runoff
    
    Parameters:
    - results (pd.DataFrame): The results from the model run, make sure you have the following columns: 'Precip', 'ET', 'Snow_melt', 'Q_s', 'Q_gw'
    - title (str): The title of the plot, if empty, no title will be shown
    - output_destination (str): The path to the output file, if empty, the plot will not be saved
    - figsize (tuple): The size of the figure, default is (12, 12)
    - fontsize (int): The fontsize of the plot, default is 12
    """

    # Some style settings, this is what I like, but feel free to change it
    sns.set_context('paper')

    # Prepare the data
    results_filtered = results.copy()
    results_filtered['Total_Runoff'] = results_filtered['Q_s'] + results_filtered['Q_gw']
    results_filtered['Month'] = results_filtered.index.month
    results_filtered['Year'] = results_filtered.index.year

    # This is a bit of a hack to get monthly sums. I'm sure there is a better way to do this
    monthly_sums = results_filtered.groupby(['Year', 'Month']).sum().reset_index()

    months = {
        1: 'Jan',
        2: 'Feb',
        3: 'Mar',
        4: 'Apr',
        5: 'May',
        6: 'Jun',
        7: 'Jul',
        8: 'Aug',
        9: 'Sep',
        10: 'Oct',
        11: 'Nov',
        12: 'Dec'
    }

    monthly_sums['Month'] = monthly_sums['Month'].map(months)

    fig = plt.figure(figsize=figsize)
    layout = (2, 2) # 2 rows, 2 columns

    # Defining the location of the subplots. This is a 2x2 grid, change the layout variable if you want to change the grid
    ax_precip = plt.subplot2grid(layout, (0, 0))
    ax_et = plt.subplot2grid(layout, (0, 1))
    ax_snow_melt = plt.subplot2grid(layout, (1, 0))
    ax_runoff = plt.subplot2grid(layout, (1, 1))

    sns.boxplot(x='Month', y='Precip', data=monthly_sums, ax=ax_precip, color=palette[0])
    sns.boxplot(x='Month', y='ET', data=monthly_sums, ax=ax_et, color=palette[1])
    sns.boxplot(x='Month', y='Snow_melt', data=monthly_sums, ax=ax_snow_melt, color=palette[2])
    sns.boxplot(x='Month', y='Total_Runoff', data=monthly_sums, ax=ax_runoff, color=palette[3])

    # Some more style settings. I recommend keeping this
    ax_precip.set_xlabel('')
    ax_precip.set_ylabel('Precipitation [mm/d]', fontsize=fontsize)
    ax_precip.tick_params(which='both', length=10, width=2, labelsize=fontsize)
    ax_precip.set_title('Monthly Precipitation', fontsize=fontsize)
    ax_precip.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    ax_et.set_xlabel('')
    ax_et.set_ylabel('Actual ET [mm/d]', fontsize=fontsize)
    ax_et.tick_params(which='both', length=10, width=2, labelsize=fontsize)
    ax_et.set_title('Monthly Actual ET', fontsize=fontsize)
    ax_et.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    ax_snow_melt.set_xlabel('')
    ax_snow_melt.set_ylabel('Snowmelt [mm/d]', fontsize=fontsize)
    ax_snow_melt.tick_params(which='both', length=10, width=2, labelsize=fontsize)
    ax_snow_melt.set_title('Monthly Snowmelt', fontsize=fontsize)
    ax_snow_melt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    ax_runoff.set_xlabel('')
    ax_runoff.set_ylabel('Total Runoff [mm/d]', fontsize=fontsize)
    ax_runoff.tick_params(which='both', length=10, width=2, labelsize=fontsize)
    ax_runoff.set_title('Monthly Total Runoff', fontsize=fontsize)
    ax_runoff.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    sns.despine()

    if title:
        plt.suptitle(title, fontsize=fontsize)

    # Save the plot if an output destination is provided
    if output_destination:
        fig.savefig(output_destination, dpi=300, bbox_inches='tight')


def plot_timeseries(results: pd.DataFrame, observed: pd.DataFrame, start_year: str, end_year: str, monthly: bool = False, title: str = '', output_destination: str = '', figsize: tuple[int, int] = (10, 6), fontsize: int = 12, palette: list[str, str] = ['#007A9A', '#25A18E']) -> None:
    """This function plots the timeseries of the observed and simulated total runoff (Q) values.

    Parameters:
    - results (pd.DataFrame): The results from the model run
    - observed (pd.DataFrame): The observed data. Should contain the following column: 'Q' for the observed runoff
    - start_year (str): The start date of the plot
    - end_year (str): The end date of the plot. The plot will be inclusive of this date
    - monthly (bool): If True, the plot will be monthly, default is False, which means the plot will be daily
    - title (str): The title of the plot, if empty, no title will be shown
    - output_destination (str): The path to the output file, if empty, the plot will not be saved
    - figsize (tuple): The size of the figure, default is (10, 6)
    - fontsize (int): The fontsize of the plot, default is 12
    """

    # Some style settings, this is what I like, but feel free to change it
    sns.set_context('paper')

    # Prepare the data
    results_filtered = results.copy()
    results_filtered = results_filtered[start_year:end_year]
    results_filtered['Total_Runoff'] = results_filtered['Q_s'] + results_filtered['Q_gw']

    observed_filtered = observed.copy()
    observed_filtered = observed_filtered[start_year:end_year]

    fig, ax = plt.subplots(figsize=figsize)

    # Resample the data if monthly is True
    if monthly:
        results_filtered = results_filtered.resample('ME').sum()
        observed_filtered = observed_filtered.resample('ME').sum()

    sns.lineplot(data=results_filtered['Total_Runoff'], ax=ax, color=palette[0], label='Simulated total runoff', alpha=0.7)
    sns.lineplot(data=observed_filtered['Q'], ax=ax, color=palette[1], label='Observed total runoff', alpha=0.7)

    ax.set_xlabel('')

    if monthly:
        ax.set_ylabel('Total runoff [mm/month]', fontsize=fontsize)
    else:
        ax.set_ylabel('Total runoff [mm/d]', fontsize=fontsize)

    ax.tick_params(which='both', length=10, width=2, labelsize=fontsize)
    ax.legend(fontsize=fontsize, loc='best')
    plt.tight_layout()
    sns.despine()
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    if title:
        plt.title(title)

    if output_destination:
        fig.savefig(output_destination, dpi=300, bbox_inches='tight')

def plot_parameter_kde(n_fold_results: pd.DataFrame, bounds: dict, output_destination: str = None, figsize: tuple[int, int] = (10, 6), fontsize: int = 12, plot_type: str = 'histplot') -> None:
    """This function plots the histogram of the parameters.
    
    Parameters:
    - n_fold_results (pd.DataFrame): The n_fold_results from the model calibration
    - bounds (dict): The bounds of the parameters. They are used to set the x-axis limits
    - output_destination (str): The path to the output file
    - figsize (tuple): The size of the figure, default is (10, 6)
    - fontsize (int): The fontsize of the plot, default is 12
    - plot_type (str): The type of plot to use. Can be either 'histplot' or 'kdeplot', default is 'histplot'
    """

    # Some style settings, this is what I like, but feel free to change it
    sns.set_context('paper')

    # Prepare the data
    n_fold_results_filtered = n_fold_results.copy()

    fig = plt.figure(figsize=figsize)
    layout = (2, 3) 

    ax_k = plt.subplot2grid(layout, (0, 0))
    ax_S_max = plt.subplot2grid(layout, (0, 1))
    ax_fr = plt.subplot2grid(layout, (0, 2))
    ax_rg = plt.subplot2grid(layout, (1, 0))
    ax_gauge_adj = plt.subplot2grid(layout, (1, 1))

    if plot_type == 'histplot':
        bins = int(np.sqrt(len(n_fold_results_filtered)))

        sns.histplot(data=n_fold_results_filtered['k'], ax=ax_k, color='#007A9A', bins=bins)
        sns.histplot(data=n_fold_results_filtered['S_max'], ax=ax_S_max, color='#007A9A', bins=bins)
        sns.histplot(data=n_fold_results_filtered['fr'], ax=ax_fr, color='#007A9A', bins=bins)
        sns.histplot(data=n_fold_results_filtered['rg'], ax=ax_rg, color='#007A9A', bins=bins)
        sns.histplot(data=n_fold_results_filtered['gauge_adj'], ax=ax_gauge_adj, color='#007A9A', bins=bins)

    elif plot_type == 'kdeplot':
        sns.kdeplot(data=n_fold_results_filtered['k'], ax=ax_k, color='#007A9A', fill=True)
        sns.kdeplot(data=n_fold_results_filtered['S_max'], ax=ax_S_max, color='#007A9A', fill=True)
        sns.kdeplot(data=n_fold_results_filtered['fr'], ax=ax_fr, color='#007A9A', fill=True)
        sns.kdeplot(data=n_fold_results_filtered['rg'], ax=ax_rg, color='#007A9A', fill=True)
        sns.kdeplot(data=n_fold_results_filtered['gauge_adj'], ax=ax_gauge_adj, color='#007A9A', fill=True)

    ax_k.set_xlabel('k', fontsize=fontsize)
    ax_k.set_ylabel('Density', fontsize=fontsize)
    ax_k.tick_params(which='both', length=10, width=2, labelsize=fontsize)
    ax_k.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax_k.set_xlim(bounds['k'])

    ax_S_max.set_xlabel('S$_{max}$', fontsize=fontsize)
    ax_S_max.set_ylabel('Density', fontsize=fontsize)
    ax_S_max.tick_params(which='both', length=10, width=2, labelsize=fontsize)
    ax_S_max.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax_S_max.set_xlim(bounds['S_max'])

    ax_fr.set_xlabel('fr', fontsize=fontsize)
    ax_fr.set_ylabel('Density', fontsize=fontsize)
    ax_fr.tick_params(which='both', length=10, width=2, labelsize=fontsize)
    ax_fr.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax_fr.set_xlim(bounds['fr'])

    ax_rg.set_xlabel('rg', fontsize=fontsize)
    ax_rg.set_ylabel('Density', fontsize=fontsize)
    ax_rg.tick_params(which='both', length=10, width=2, labelsize=fontsize)
    ax_rg.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax_rg.set_xlim(bounds['rg'])

    ax_gauge_adj.set_xlabel('gauge_adj', fontsize=fontsize)
    ax_gauge_adj.set_ylabel('Density', fontsize=fontsize)
    ax_gauge_adj.tick_params(which='both', length=10, width=2, labelsize=fontsize)
    ax_gauge_adj.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax_gauge_adj.set_xlim(bounds['gauge_adj'])

    plt.tight_layout()
    sns.despine()

    if output_destination:
        fig.savefig(output_destination, dpi=300, bbox_inches='tight')