import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Optional


def preprocess_data(
    path_to_file: str,
    catchment_area: float,
    output_destination: str = "",
    start_year: int = 1986,
    filter_dates: Optional[Tuple[int, int]] = None,
) -> pd.DataFrame:
    """This function takes the .txt file, transforms it into a pandas DataFrame, and optionally filters the data.

    Args:
        path_to_file (str): The path to the .txt file
        catchment_area (float): The catchment area in km^2
        output_destination (str): The path to the new .csv file
        start_year (int): The year to start the data from. Default is 1986.
        filter_dates (Optional[Tuple[int, int]]): A tuple of (start_year, end_year) to filter the data. Default is None.

    Returns:
        pd.DataFrame: The DataFrame containing the processed and filtered data.
    """

    # Read the file, skipping the header lines
    df = pd.read_csv(path_to_file, sep="\s+", skiprows=1)

    # Extract the start date from the file header
    with open(path_to_file, "r") as file:
        header = file.readline().strip()

    # Parse the date range from the header
    date_range = header.split(": ")[1].split(" - ")[0]
    file_start_date = datetime.strptime(date_range, "%d/%m/%Y")

    # Convert index to datetime
    def day_to_date(day):
        return file_start_date + timedelta(days=int(day))

    # Apply the conversion to create a new 'date' column
    df["date"] = df.index.map(day_to_date)

    # Set 'date' as the index
    df.set_index("date", inplace=True)

    # Convert all columns to numeric, replacing any non-numeric values with NaN
    df = df.apply(pd.to_numeric, errors="coerce")

    # Filter the DataFrame based on start_year
    df = df.loc[f"{start_year}":]

    # Apply additional filtering if filter_dates is provided
    if filter_dates:
        filter_start, filter_end = filter_dates
        df = df.loc[f"{filter_start}":f"{filter_end}"]

    # Convert Q from m^3/s to mm/day
    if "Q" in df.columns:
        df["Q"] = (df["Q"] * 60 * 60 * 24) / catchment_area / 1000

    if output_destination:
        df.to_csv(output_destination, index=True, header=True)

    return df


def train_validate_split(
    data: pd.DataFrame, train_size: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """This function splits the data into training and validating sets.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        train_size (float): The proportion of the data to use for training. This is a value between 0 and 1.

    Returns:
        tuple: A tuple containing the training and testing DataFrames.
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

    path_to_file = "/Users/cooper/Desktop/bucket-model/data/GSTEIGmeteo.txt"
    output_destination = "/Users/cooper/Desktop/bucket-model/data/GSTEIGmeteo.csv"
    catchment_area = 384.2  # km^2
    data = preprocess_data(path_to_file, output_destination, catchment_area)
    print(data)


if __name__ == "__main__":
    main()
