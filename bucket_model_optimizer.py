import pandas as pd
from scipy.optimize import minimize, basinhopping
import numpy as np
from dataclasses import dataclass, field
from bucket_model import BucketModel
from metrics import rmse, nse, log_nse, mae, kge, pbias
import concurrent.futures
from typing import Union
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# If you want to add a new metric, you need to implement it in metrics.py and add it to the GOF_DICT dictionary.
GOF_DICT = {
    'rmse': rmse,
    'nse': nse,
    'log_nse': log_nse,
    'mae': mae,
    'kge': kge,
    'pbias': pbias
}

@dataclass
class BucketModelOptimizer():
    model: BucketModel
    training_data: pd.DataFrame
    validation_data: pd.DataFrame = None

    method: str = field(init=False, repr=False)
    bounds: dict = field(init=False, repr=False)
    folds: int = field(init=False, repr=False)

    @staticmethod
    def create_param_dict(keys, values):
        """This is a helper function that creates a dictionary from two lists."""
        return {key: value for key, value in zip(keys, values)}
    

    def set_options(self, method: str, bounds: dict, folds: int = 0) -> None:
        """
        This method sets the optimization method and bounds for the calibration.

        Parameters:
        - method (str): The optimization method to use. Can be either 'local' or 'global'.
        - bounds (dict): A dictionary containing the lower and upper bounds for each parameter.
        """
        possible_methods = ['local', 'n-folds', 'global']

        if method not in possible_methods:
            raise ValueError(f"Method must be one of {possible_methods}")
        
        self.method = method
        self.bounds = bounds

        if method == 'n-folds' and folds == 0:
            raise ValueError("You must provide the number of folds for the n-folds method.")
        self.folds = folds

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

        simulated_Q = results['Q_s'] + results['Q_gw']

        # Objective function is RMSE, minimized. Change metric if needed, adjust sign accordingly.
        objective_function = rmse(simulated_Q, self.training_data['Q']) 

        return objective_function
    
    def single_fold_calibration(self, bounds_list: list[tuple], initial_guess: list[float] = None) -> list[float]:
        """Performs a single fold calibration using random initial guesses.
        
        Parameters:
        - bounds_list (list[tuple]): A list of tuples containing the lower and upper bounds for each parameter.
        - initial_guess (list[float]): A list of initial guesses for the parameters"""

        if initial_guess is None:
            initial_guess = [round(np.random.uniform(lower, upper), 3) for lower, upper in bounds_list] # Round to 3 decimal places

        options = {
                'eps': 1e-3, 
            }
        result = minimize(
            lambda params: self._objective_function(params),
            initial_guess,
            method='L-BFGS-B', # Have a look at the doc for more methods: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
            bounds=bounds_list,
            options=options
        )
        return [round(param, 3) for param in result.x]


    def calibrate(self, initial_guess : list[float] = None) -> tuple[dict, pd.DataFrame]:
        """
        This method calibrates the model's parameters using the method and bounds
        specified in the set_options method. The method can be either 'local', 'global' or 'n-folds'.

        Parameters:
        - initial_guess (list[float]): A list of initial guesses for the parameters. If no initial guesses are provided, uniform random values are sampled from the bounds.

        Returns:
        - tuple[dict, pd.DataFrame]: A tuple containing the calibrated parameters and the results of the n-folds calibration. If the method is 'local' or 'global', the second element is None.
        """

        # This is a list of tuples. Each tuple contains the lower and upper bounds for each parameter.
        bounds_list = list(self.bounds.values())

        # Randomly generate an initial guess for each parameter.
        if self.method == 'local':

            optimal_param_list = self.single_fold_calibration(bounds_list, initial_guess)
            calibrated_parameters = BucketModelOptimizer.create_param_dict(self.bounds.keys(), optimal_param_list)

            self.model.update_parameters(calibrated_parameters)

            return calibrated_parameters, None
        
        elif self.method == 'n-folds':
            with concurrent.futures.ProcessPoolExecutor() as executor:
                n_fold_results = list(executor.map(self.single_fold_calibration, [bounds_list] * self.folds))

            columns = list(self.bounds.keys())
            n_fold_results = pd.DataFrame(n_fold_results, columns=columns)
            # print(results)

            calibrated_parameters = self.get_best_parameters(n_fold_results)

            return calibrated_parameters, n_fold_results

        # # TODO: Get rid of this?
        # elif self.method == 'global':
        #     # The way basinhopping works, is that it find a bunch of local minima and then chooses the best one. Hence the name 'basinhopping'. 
        #     # The method is slow (many hours). I do not recommend using it unless you have a good reason to do so. 
        #     # You can try to tune the parameters of the method to make it faster: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html
        #     result = basinhopping(
        #         lambda params: self._objective_function(params),
        #         initial_guess,
        #         minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': bounds_list}
        #     )

        #     optimal_param_list = [round(param, 3) for param in result.x]
        #     calibrated_parameters = BucketModelOptimizer.create_param_dict(self.bounds.keys(), optimal_param_list)

        # # Update the model's parameters with the calibrated parameters. This is important for scoring the model.
        # self.model.update_parameters(calibrated_parameters)

        # return calibrated_parameters

    def get_best_parameters(self, results: pd.DataFrame) -> dict:
        """This function takes a DataFrame containing the results of the n-folds calibration and returns the one that performs best.
        
        Parameters:
        - results (pd.DataFrame): A DataFrame containing the results of the n-folds calibration.
        
        Returns:
        - dict: A dictionary containing the best parameters.
        """
        best_rmse = float('inf')
        best_parameters = None

        for index, row in results.iterrows():
            # Convert row to parameter dictionary
            params = row.to_dict()
            
            self.model.update_parameters(params)
            
            simulated_results = self.model.run(self.training_data)
            
            simulated_Q = simulated_results['Q_s'] + simulated_results['Q_gw']
            observed_Q = self.training_data['Q']
            
            # Calculate RMSE
            current_rmse = rmse(simulated_Q, observed_Q)
            
            # Check if the current RMSE is the best one
            if current_rmse < best_rmse:
                best_rmse = current_rmse
                best_parameters = params

        # Update the model's parameters with the best parameters, otherwise the last set of parameters will be used.
        self.model.update_parameters(best_parameters)

        return best_parameters

    def score_model(self, metrics: list[str] = ['rmse']) -> dict:
        """
        This function calculates the goodness of fit metrics for a given model.

        Parameters:
        - metrics (list(str)): A list of strings containing the names of the metrics to calculate. If no metrics are provided, only RMSE is calculated.

        Returns:
        - dict: A dictionary containing the scores for the training and validation data.
        """

        metrics = [metric.lower() for metric in metrics] # Convert all metrics to lowercase

        training_results = self.model.run(self.training_data)
        simulated_Q = training_results['Q_s'] + training_results['Q_gw']
        observed_Q = self.training_data['Q']
        training_score = {metric: round(GOF_DICT[metric](simulated_Q, observed_Q), 3) for metric in metrics}

        scores = {'training': training_score}

        if self.validation_data is not None:
            validation_results = self.model.run(self.validation_data)
            simulated_Q = validation_results['Q_s'] + validation_results['Q_gw']
            observed_Q = self.validation_data['Q']
            validation_score = {metric: round(GOF_DICT[metric](simulated_Q, observed_Q), 3) for metric in metrics}

            scores['validation'] = validation_score

        return scores
    
    # TODO: Add customization options after meeting with team
    def plot_of_surface(self, param1: str, param2: str, n_points: int) -> None:
        """
        This function creates a 2D plot of the objective function surface for two parameters.

        Parameters:
        - param1 (str): The name of the first parameter.
        - param2 (str): The name of the second parameter.
        - n_points (int): The number of points to sample for each parameter.
        """
        params = self.model.get_parameters().copy()
        # print(params)
        param1_values = np.linspace(self.bounds[param1][0], self.bounds[param1][1], n_points)
        param2_values = np.linspace(self.bounds[param2][0], self.bounds[param2][1], n_points)
        PARAM1, PARAM2 = np.meshgrid(param1_values, param2_values)

        goal_matrix = np.zeros(PARAM1.shape)

        # Compute the objective function for each combination of param1 and param2
        for i in range(n_points):
            for j in range(n_points):
                params_copy = params.copy()
                params_copy[param1] = PARAM1[i, j]
                params_copy[param2] = PARAM2[i, j]
                goal_matrix[i, j] = self._objective_function(list(params_copy.values()))

        # Plotting the surface
        plt.figure(figsize=(10, 7))
        levels = np.linspace(np.min(goal_matrix), np.max(goal_matrix), 20)

        CP = plt.contour(PARAM1, PARAM2, goal_matrix, levels=levels, cmap='viridis')
        plt.clabel(CP, inline=True, fontsize=10)

        plt.xlabel(f'{param1} (mm/d/°C)')
        plt.ylabel(f'{param2} (days)')

        # print(params)

        plt.scatter(params[param1], params[param2], color='red', label='Optimal Point')
        plt.legend()
        plt.show()




 



