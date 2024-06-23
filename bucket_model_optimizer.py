import pandas as pd
from scipy.optimize import minimize, basinhopping
import numpy as np
from dataclasses import dataclass, field
from bucket_model import BucketModel
from metrics import nse, log_nse, mae, kge, pbias, rmse
from concurrent.futures import ThreadPoolExecutor
from typing import Union
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from functools import partial

# If you want to add a new metric, you need to implement it in metrics.py and add it to the GOF_DICT dictionary.
GOF_DICT = {
    "rmse": rmse,
    "nse": nse,
    "log_nse": log_nse,
    "mae": mae,
    "kge": kge,
    "pbias": pbias,
}


@dataclass
class BucketModelOptimizer:
    """
    A class to optimize the parameters of a BucketModel.

    Parameters:
    - model (BucketModel): The bucket model instance to be optimized.
    - training_data (pd.DataFrame): DataFrame containing the training data with columns 'P_mix', 'T_max', 'T_min', and 'Q'.
    - validation_data (pd.DataFrame, optional): DataFrame containing the validation data with columns 'P_mix', 'T_max', 'T_min', and 'Q'.

    Attributes:
    - method (str): The optimization method to be used ('local', 'global', or 'n-folds').
    - bounds (dict): Dictionary containing the lower and upper bounds for each parameter.
    - folds (int): Number of folds for n-folds cross-validation.

    Methods:
    - create_param_dict: Helper function to create a dictionary from two lists of keys and values.
    - set_options: Set the optimization method, bounds, and number of folds.
    - _objective_function: Calculate the objective function (NSE) for the optimization algorithm.
    - single_fold_calibration: Perform a single fold calibration using random initial guesses.
    - calibrate: Calibrate the model's parameters using the specified method and bounds.
    - get_best_parameters: Retrieve the best parameters from the calibration results.
    - score_model: Calculate goodness of fit metrics for the training and validation data.
    - plot_of_surface: Create a 2D plot of the objective function surface for two parameters.
    """

    model: BucketModel
    training_data: pd.DataFrame
    validation_data: pd.DataFrame = None

    _model_copy: BucketModel = field(init=False, repr=False)
    method: str = field(init=False, repr=False)
    bounds: dict = field(init=False, repr=False)
    folds: int = field(default=1, init=False, repr=False)

    def __post_init__(self):
        self._model_copy = self.model.copy()

    @staticmethod
    def create_param_dict(keys: list, values: list) -> dict:
        """This is a helper function that creates a dictionary from two lists.

        Parameters:
        - keys (list): A list of keys.
        - values (list): A list of values.

        Returns:
        - dict: A dictionary containing the keys and values."""
        return {key: value for key, value in zip(keys, values)}

    def set_options(self, method: str, bounds: dict, folds: int = 1) -> None:
        """
        This method sets the optimization method and bounds for the calibration.

        Parameters:
        - method (str): The optimization method to use. Can be either 'local' or 'global'.
        - bounds (dict): A dictionary containing the lower and upper bounds for each parameter.
        """
        possible_methods = ["local", "n-folds"]

        if method not in possible_methods:
            raise ValueError(f"Method must be one of {possible_methods}")

        if method == "n-folds" and folds == 1:
            raise ValueError(
                "You must provide the number of folds for the n-folds method."
            )
        self.folds = folds

        self.method = method
        self.bounds = bounds

    def _objective_function(self, params: list) -> float:
        """
        This is a helper function that calculates the objective function for the optimization algorithm.

        Parameters:
        - params (list): A list of parameters to calibrate.

        Returns:
        - float: The value of the objective function.
        """
        model_copy = self.model.copy()

        # Create a dictionary from the parameter list. Look like this {'parameter_name': value, ...}
        param_dict = BucketModelOptimizer.create_param_dict(self.bounds.keys(), params)

        model_copy.update_parameters(param_dict)

        results = model_copy.run(self.training_data)

        simulated_Q = results["Q_s"] + results["Q_gw"]

        # Objective function is NSE, minimized. Change metric if needed, adjust sign accordingly.
        objective_function = -nse(simulated_Q, self.training_data["Q"])

        return round(objective_function, 6)

    def single_fold_calibration(
        self,
        bounds_list: list[tuple],
        initial_guess: list[float] = None,
        verbose: bool = False,
    ) -> list[float]:
        """Performs a single fold calibration using random initial guesses.

        Parameters:
        - bounds_list (list[tuple]): A list of tuples containing the lower and upper bounds for each parameter.
        - initial_guess (list[float]): A list of initial guesses for the parameters
        - verbose (bool): A boolean indicating whether to print the current parameter values at each iteration.
        """

        if initial_guess is None:
            initial_guess = [
                round(np.random.uniform(lower, upper), 6)
                for lower, upper in bounds_list
            ]

        self._model_copy.update_parameters(
            self.create_param_dict(self.bounds.keys(), initial_guess)
        )

        if verbose:
            print(f"Initial guess: {initial_guess}")

        options = {
            "ftol": 1e-5,
            "gtol": 1e-5,
            "eps": 1e-3,
        }

        def print_status(x):
            if verbose:
                print("Current parameter values:", np.round(x, 2))

        result = minimize(
            self._objective_function,
            initial_guess,
            method="L-BFGS-B",  # Have a look at the doc for more methods: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
            bounds=bounds_list,
            options=options,
            jac=None,
            callback=print_status if verbose else None,
        )
        return [round(param, 3) for param in result.x]

    def calibrate(
        self, initial_guess: list[float] = None, verbose: bool = False
    ) -> tuple[dict, pd.DataFrame]:
        """
        This method calibrates the model's parameters using the method and bounds
        specified in the set_options method. The method can be either 'local' or 'n-folds'.

        Parameters:
        - initial_guess (list[float]): A list of initial guesses for the parameters. If no initial guesses are provided, uniform random values are sampled from the bounds.
        - verbose (bool): A boolean indicating whether to print the current parameter values at each iteration.

        Returns:
        - tuple[dict, pd.DataFrame]: A tuple containing the calibrated parameters and the results of the n-folds calibration. If the method is 'local' or 'global', the second element is None.

        """
        # This is a list of tuples. Each tuple contains the lower and upper bounds for each parameter.
        bounds_list = list(self.bounds.values())

        with ThreadPoolExecutor() as executor:
            calibration_results = list(
                executor.map(
                    self.single_fold_calibration,
                    [bounds_list] * self.folds,
                    [None] * self.folds,
                    [verbose] * self.folds,
                )
            )

        columns = list(self.bounds.keys())
        calibration_results = pd.DataFrame(calibration_results, columns=columns)

        calibrated_parameters = self.get_best_parameters(calibration_results)

        return calibrated_parameters, calibration_results

    def get_best_parameters(self, results: pd.DataFrame) -> dict:
        """This function takes a DataFrame containing the results of the n-folds calibration and returns the one that performs best.

        Parameters:
        - results (pd.DataFrame): A DataFrame containing the results of the n-folds calibration.

        Returns:
        - dict: A dictionary containing the best parameters.
        """
        best_nse = float("inf")
        best_parameters = None
        model_copy = self._model_copy.copy()

        for index, row in results.iterrows():
            params = row.to_dict()
            model_copy.update_parameters(params)
            simulated_results = model_copy.run(self.training_data)
            simulated_Q = simulated_results["Q_s"] + simulated_results["Q_gw"]
            observed_Q = self.training_data["Q"]
            current_nse = nse(simulated_Q, observed_Q)

            if current_nse < best_nse:
                best_nse = current_nse
                best_parameters = params

        self._model_copy.update_parameters(best_parameters)
        return best_parameters

    def score_model(self, metrics: list[str] = ["nse"]) -> dict:
        """
        This function calculates the goodness of fit metrics for a given model.

        Parameters:
        - metrics (list(str)): A list of strings containing the names of the metrics to calculate. If no metrics are provided, only nse is calculated.

        Returns:
        - dict: A dictionary containing the scores for the training and validation data.
        """

        metrics = [metric.lower() for metric in metrics]

        training_results = self._model_copy.run(self.training_data)
        simulated_Q = training_results["Q_s"] + training_results["Q_gw"]
        observed_Q = self.training_data["Q"]
        training_score = {
            metric: round(GOF_DICT[metric](simulated_Q, observed_Q), 3)
            for metric in metrics
        }

        scores = {"training": training_score}

        if self.validation_data is not None:
            validation_results = self._model_copy.run(self.validation_data)
            simulated_Q = validation_results["Q_s"] + validation_results["Q_gw"]
            observed_Q = self.validation_data["Q"]
            validation_score = {
                metric: round(GOF_DICT[metric](simulated_Q, observed_Q), 3)
                for metric in metrics
            }
            scores["validation"] = validation_score

        return scores

    def plot_of_surface(
        self,
        param1: str,
        param2: str,
        n_points: int,
        unit_1: str,
        unit_2: str,
        figsize: tuple[int, int] = (10, 6),
        fontsize: int = 12,
        cmap: str = "viridis",
        decimal_places: int = 2,
    ) -> None:
        """
        This function creates a 2D plot of the objective function surface for two parameters.

        Parameters:
        - param1 (str): The name of the first parameter.
        - param2 (str): The name of the second parameter.
        - n_points (int): The number of points to sample for each parameter.
        - unit_1 (str): The unit of the first parameter.
        - unit_2 (str): The unit of the second parameter.
        - figsize (tuple): The size of the figure.
        - fontsize (int): The font size of the labels.
        - cmap (str): The color map to use for the contour plot.
        - decimal_places (int): The number of decimal places for the contour labels.
        """

        model_params = self._model_copy.get_parameters()

        if param1 not in model_params:
            raise ValueError(f"Parameter '{param1}' does not exist in the model.")

        if param2 not in model_params:
            raise ValueError(f"Parameter '{param2}' does not exist in the model.")

        if param1 not in self.bounds:
            raise ValueError(f"Bounds for parameter '{param1}' are not defined.")

        if param2 not in self.bounds:
            raise ValueError(f"Bounds for parameter '{param2}' are not defined.")

        params = model_params.copy()

        param1_values = np.linspace(
            self.bounds[param1][0], self.bounds[param1][1], n_points
        )
        param2_values = np.linspace(
            self.bounds[param2][0], self.bounds[param2][1], n_points
        )
        PARAM1, PARAM2 = np.meshgrid(param1_values, param2_values)

        goal_matrix = np.zeros(PARAM1.shape)

        # Calculate the objective function for each combination of parameters
        for i in range(n_points):
            for j in range(n_points):

                params_copy = params.copy()

                params_copy[param1] = PARAM1[i, j]
                params_copy[param2] = PARAM2[i, j]

                goal_matrix[i, j] = -self._objective_function(
                    list(params_copy.values())
                )

        plt.figure(figsize=figsize)
        levels = np.linspace(np.min(goal_matrix), np.max(goal_matrix), 20)

        CP = plt.contour(PARAM1, PARAM2, goal_matrix, levels=levels, cmap=cmap)
        plt.clabel(CP, inline=True, fontsize=10, fmt=f"%.{decimal_places}f")

        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        plt.xlabel(f"{param1} [{unit_1}]", fontsize=fontsize)
        plt.ylabel(f"{param2} [{unit_2}]", fontsize=fontsize)

        sns.despine()

        plt.scatter(params[param1], params[param2], color="red", label="Optimal Point")
        plt.legend()
        plt.show()

    def local_sensitivity_analysis(
        self,
        percent_change: float = 5,
    ) -> pd.DataFrame:
        """
        Perform local sensitivity analysis on the model parameters.

        Parameters:
        - percent_change (float): The percentage change to apply to the parameters.

        Returns:
        - pd.DataFrame: A DataFrame summarizing the sensitivity analysis results.
        """

        def compute_annual_runoff(model: BucketModel) -> float:
            results = model.run(self.training_data)
            annual_runoff = results["Q_s"] + results["Q_gw"]
            return annual_runoff.mean()

        original_params = self._model_copy.get_parameters().copy()
        original_runoff = compute_annual_runoff(self._model_copy)

        sensitivity_results = []

        for param in original_params.keys():
            param_value = original_params[param]

            params_plus = original_params.copy()

            params_plus[param] = param_value * (1 + percent_change / 100)

            self._model_copy.update_parameters(params_plus)
            runoff_plus = compute_annual_runoff(self._model_copy)

            params_minus = original_params.copy()

            params_minus[param] = param_value * (1 - percent_change / 100)

            self._model_copy.update_parameters(params_minus)
            runoff_minus = compute_annual_runoff(self._model_copy)

            delta_P_plus = param_value * (percent_change / 100)
            delta_P_minus = -param_value * (percent_change / 100)

            sensitivity_plus = round(
                ((runoff_plus - original_runoff) / delta_P_plus)
                * (param_value / original_runoff),
                4,
            )
            sensitivity_minus = round(
                ((runoff_minus - original_runoff) / delta_P_minus)
                * (param_value / original_runoff),
                4,
            )

            sensitivity_results.append(
                {
                    "Parameter": param,
                    f"Sensitivity +{percent_change}%": sensitivity_plus,
                    f"Sensitivity -{percent_change}%": sensitivity_minus,
                }
            )

        self._model_copy.update_parameters(original_params)

        return pd.DataFrame(sensitivity_results)

    def sync_models(self, direction: str = "to_original") -> None:
        """
        Synchronize the working copy and the original model.

        Parameters:
        - direction (str): The direction of synchronization.
                           'to_original': Apply changes from working copy to original model (default).
                           'from_original': Reset working copy to match the original model.

        Raises:
        - ValueError: If an invalid direction is provided.
        """
        if direction == "to_original":
            self.model.update_parameters(self._model_copy.get_parameters())
            print("Changes from working copy applied to the original model.")
        elif direction == "from_original":
            self._model_copy = self.model.copy()
            print("Working copy reset to match the original model.")
        else:
            raise ValueError("Invalid direction. Use 'to_original' or 'from_original'.")
        

    def get_optimized_model(self) -> BucketModel:
        """
        Synchronize the working copy with the original model and return a copy of the optimized model.

        Returns:
        - BucketModel: A copy of the optimized model.
        """
        self.sync_models('to_original')
        return self.model.copy()
