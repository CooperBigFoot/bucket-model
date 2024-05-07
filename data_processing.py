import pandas as pd 

def preprocess_data(path_to_file: str, catchment_area: float, output_destination: str = '') -> None:
    """This function takes the .txt file you find on moodle and transforms it into a pandas DataFrame.
    
    Parameters:
    - path_to_file (str): The path to the .txt file
    - output_destination (str): The path to the new .csv file
    - catchment_area (float): The catchment area in km^2
    """

    precipitation = pd.read_csv(path_to_file, sep=r'\s+', skiprows=1, header=0)

    # Create a DatetimeIndex starting from October 1st, 1985
    date_range = pd.date_range(start='1986-01-01', periods=len(precipitation), freq='D')

    # Set the DatetimeIndex as the new index of the DataFrame
    precipitation.set_index(date_range, inplace=True)

    # Rename index to 'date'
    precipitation.index.name = 'date'

    # Set date to datetime format
    precipitation.index = pd.to_datetime(precipitation.index)
    precipitation = precipitation.apply(pd.to_numeric, errors='coerce')

    # Only keep data from 1986  to end of 1999
    precipitation = precipitation.loc['1986':'1999']

    precipitation['Q'] = (precipitation['Q'] * 60 * 60 * 24) / catchment_area / 1000 # Convert m^3/s to mm/day
    
    if output_destination:
        precipitation.to_csv(output_destination, index=True, header=True)
    
    return precipitation

def train_validate_split(data: pd.DataFrame, train_size: float) -> tuple:
    """This function splits the data into training and validating sets.
    
    Parameters:
    - data (pd.DataFrame): The DataFrame containing the data.
    - train_size (float): The proportion of the data to use for training. This is a value between 0 and 1.
    
    Returns:
    - tuple: A tuple containing the training and testing DataFrames.
    """
    
    train_size = int(len(data) * train_size)
    train_data = data.iloc[:train_size]
    validate_data = data.iloc[train_size:]
    
    return train_data, validate_data        



def main() -> None:
    """
    This is an example of how you can use the preprocess_data function. You need to change the path_to_file and output_destination to your own paths.
    Alternatively you can import this function into another script and use it there. See example_run.ipynb for more information.
    """

    path_to_file = '/Users/cooper/Desktop/bucket-model/data/GSTEIGmeteo.txt'
    output_destination = '/Users/cooper/Desktop/bucket-model/data/GSTEIGmeteo.csv'
    catchment_area = 384.2 # km^2
    data = preprocess_data(path_to_file, output_destination, catchment_area)
    print(data)
    
    
if __name__ == "__main__":
    main()

