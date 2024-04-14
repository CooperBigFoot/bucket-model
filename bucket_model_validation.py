import pandas as pd 
from bucket_model import BucketModel


def score_model(model: BucketModel, scoring_data: pd.DataFrame, metrics: list[str]) -> dict:
    """
    This function calculates the goodness of fit metrics for a given model.

    Parameters:
    - model (BucketModel): The model to score.
    - scoring_data (pd.DataFrame): The DataFrame containing the data to score the model on, can be either training or validation data.
    - metrics (list(str)): A list of strings containing the names of the metrics to calculate.
    """

    raise NotImplementedError("This function is not implemented yet.")

