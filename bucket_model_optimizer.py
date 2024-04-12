import pandas as pd
from scipy.optimize import minimize, basinhopping
import numpy as np
from dataclasses import dataclass, field
from bucket_model import BucketModel


@dataclass
class BucketModelOptimizer:
    model: BucketModel
    training_data: pd.DataFrame
    method: str = field(init=False, repr=False)
    bounds: dict = field(init=False, repr=False)

    # You should define some other Goodness of Fit metrics here too (e.g. log_nse, mae, pbias). I only included rmse as an example.
    def rmse(self, simulated_Q: pd.DataFrame, observed_Q: pd.Series) -> float:
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


    def set_options(self, method: str, bounds: dict) -> None:
        """
        This method sets the optimization method and bounds for the calibration.

        Parameters:
        method (str): The optimization method to use. Can be either 'local' or 'global'.
        bounds (dict): A dictionary containing the lower and upper bounds for each parameter.
        """
        possible_methods = ['local', 'global']

        if method not in possible_methods:
            raise ValueError(f"Method must be one of {possible_methods}")
        
        self.method = method
        self.bounds = bounds

    def _objective_function(self, params: np.ndarray) -> float:
        """
        This is a private method used to calculate the objective function, 
        which in this case is the RMSE between observed and simulated Q values.

        The params array contains the optimization variables, which need to be
        mapped to the model's parameters.
        """
        # Map the params array to the model's parameters
        param_dict = {key: value for key,
                      value in zip(self.bounds.keys(), params)}
        self.model.update_parameters(param_dict)

        observed_Q = self.training_data['Q']
        data = self.training_data.drop(columns='Q', axis=1)

        model_run = self.model.run(data)
        simulated_Q = model_run['Q_s'] + model_run['Q_gw']
        return self.rmse(simulated_Q, observed_Q)

    def calibrate(self) -> dict:
        """
        This method optimizes the model's parameters using the method and bounds
        specified in the set_options method. The method can be either 'local' or 'global'.

        Returns:
        dict: A dictionary containing the optimal parameters.
        """

        def create_param_dict(keys, values):
            """This is a helper function that creates a dictionary from two lists."""

            return {key: value for key, value in zip(keys, values)}
        
        # This is a list of tuples. Each tuple contains the lower and upper bounds for each parameter.
        bounds_list = list(self.bounds.values())

        # Randomly generate an initial guess for each parameter.
        initial_guess = [np.random.uniform(lower, upper) for lower, upper in bounds_list]

        if self.method == 'local':
            method = 'L-BFGS-B'  # Have a look at the doc for more methods: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

            # Use a lambda to pass 'self' to the objective function
            result = minimize(
                lambda params: self._objective_function(params),
                initial_guess,
                method=method,
                bounds=bounds_list
            )

            optimal_param_list = result.x
            calibrated_parameters = create_param_dict(self.bounds.keys(), optimal_param_list)

            return calibrated_parameters
        
        elif self.method == 'global':
            # The way basinhopping works, is that it find a bunch of local minima and then chooses the best one. Hence the name 'basinhopping'. 
            # The method is slow (many hours). I do not recommend using it unless you have a good reason to do so. 
            # You can try to tune the parameters of the method to make it faster: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html
            result = basinhopping(
                lambda params: self._objective_function(params),
                initial_guess,
                minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': bounds_list}
            )

            optimal_param_list = result.x
            calibrated_parameters = create_param_dict(self.bounds.keys(), optimal_param_list)

            return calibrated_parameters
