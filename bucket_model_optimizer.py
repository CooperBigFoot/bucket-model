import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

from bucket_model import BucketModel
from metrics import nse, log_nse, mae, kge, pbias, rmse


# Dictionary mapping metric names to their functions
GOF_METRICS = {
    "rmse": rmse,        # Root Mean Squared Error (lower is better)
    "nse": nse,          # Nash-Sutcliffe Efficiency (higher is better)
    "log_nse": log_nse,  # Log Nash-Sutcliffe Efficiency (higher is better)
    "mae": mae,          # Mean Absolute Error (lower is better)
    "kge": kge,          # Kling-Gupta Efficiency (higher is better)
    "pbias": pbias,      # Percent Bias (closer to 0 is better)
}

# Metrics where higher values indicate better performance
MAXIMIZE_METRICS = {"nse", "log_nse", "kge"}


class BucketModelOptimizer:
    """
    Optimizer for calibrating and evaluating BucketModel parameters.

    This class provides methods to calibrate BucketModel parameters using various
    optimization techniques, evaluate model performance with multiple goodness-of-fit
    metrics, and analyze parameter sensitivity.

    Parameters
    ----------
    model : BucketModel
        The bucket model instance to optimize
    training_data : pd.DataFrame
        DataFrame with training data for calibration, must include columns for
        precipitation ('P_mix'), temperatures ('T_max', 'T_min'), and observed 
        discharge ('Q')
    validation_data : pd.DataFrame, optional
        DataFrame with validation data for model evaluation, must have the same
        structure as training_data

    Attributes
    ----------
    model : BucketModel
        The original model instance provided during initialization
    _model_copy : BucketModel
        Working copy of the model used during optimization to preserve the original
    training_data : pd.DataFrame
        Data used for calibrating the model
    validation_data : pd.DataFrame or None
        Data used for validating the model (if provided)
    method : str
        Current optimization method
    bounds : Dict[str, Tuple[float, float]]
        Parameter bounds for optimization
    objective_function : str
        Name of the goodness-of-fit metric used for optimization
    folds : int
        Number of optimization runs for n-folds method
    """

    def __init__(
        self,
        model: BucketModel,
        training_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None
    ):
        """
        Initialize the optimizer with a model and calibration data.
        """
        # Store the original model and create a working copy
        self.model = model
        self._model_copy = model.copy()

        # Store datasets
        self.training_data = training_data
        self.validation_data = validation_data

        # Check if training data has required columns
        required_columns = {'P_mix', 'T_max', 'T_min', 'Q'}
        missing_columns = required_columns - set(training_data.columns)
        if missing_columns:
            raise ValueError(
                f"Training data missing required columns: {missing_columns}")

        # Check validation data if provided
        if validation_data is not None:
            missing_columns = required_columns - set(validation_data.columns)
            if missing_columns:
                raise ValueError(
                    f"Validation data missing required columns: {missing_columns}")

        # Optimization configuration (will be set by set_options)
        self.method = "local"
        self.bounds = {}
        self.objective_function = "nse"
        self.folds = 5

        # Calibration results
        self.calibration_results = None
        self.best_parameters = None

        print("BucketModelOptimizer initialized successfully.")
        print(f"Training data shape: {training_data.shape}")

        if validation_data is not None:
            print(f"Validation data shape: {validation_data.shape}")
        else:
            print("No validation data provided.")

    def set_options(
        self,
        method: str = "local",
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        folds: int = 5,
        objective_function: str = "nse"
    ) -> None:
        """
        Configure optimization options for model calibration.

        Parameters
        ----------
        method : str, default="local"
            Optimization method to use. Options:
            - "local": Single optimization with one initial guess
            - "n-folds": Multiple optimizations with different initial guesses
        bounds : Dict[str, Tuple[float, float]], optional
            Parameter bounds for optimization as {param_name: (lower_bound, upper_bound)}.
            If None, default bounds will be used based on typical values.
        folds : int, default=5
            Number of optimization runs for "n-folds" method.
            Ignored if method is "local".
        objective_function : str, default="nse"
            Goodness-of-fit metric to optimize. Options:
            - "nse": Nash-Sutcliffe Efficiency (maximize)
            - "rmse": Root Mean Square Error (minimize)
            - "log_nse": Log Nash-Sutcliffe Efficiency (maximize)
            - "kge": Kling-Gupta Efficiency (maximize)
            - "mae": Mean Absolute Error (minimize)
            - "pbias": Percent Bias (minimize, absolute value)

        Raises
        ------
        ValueError
            If method is not recognized or configuration is invalid

        Notes
        -----
        When using "n-folds" method, multiple optimizations will be run with 
        different random initial parameter guesses. This helps avoid local optima
        but takes longer to compute.
        """
        # Set default bounds if not provided
        if bounds is None:
            bounds = {
                "k": (1, 1.5),             # Degree-day factor [mm/°C/day]
                "S_max": (10, 50),         # Max soil water capacity [mm]
                "fr": (0.1, 0.3),          # Impermeable area fraction [-]
                "rg": (10, 35),            # Groundwater residence time [days]
                "snow_threshold_temp": (-1.0, 4.0),  # Snow threshold [°C]
            }

        # Validate method
        valid_methods = ["local", "n-folds"]

        if method not in valid_methods:
            raise ValueError(
                f"Method must be one of {valid_methods}, got {method}")

        # Validate bounds against model parameters
        model_params = set(self.model.get_parameters().keys())
        bound_params = set(bounds.keys())

        if not bound_params.issubset(model_params):
            invalid_params = bound_params - model_params
            raise ValueError(f"Invalid parameters in bounds: {invalid_params}")

        # Validate objective function
        valid_metrics = list(GOF_METRICS.keys())
        if objective_function not in valid_metrics:
            raise ValueError(
                f"Objective function must be one of {valid_metrics}, got {objective_function}")

        # Validate folds
        if method == "n-folds" and (not isinstance(folds, int) or folds < 1):
            raise ValueError(
                f"Number of folds must be a positive integer, got {folds}")

        # Store configuration
        self.method = method
        self.bounds = bounds
        self.folds = folds
        self.objective_function = objective_function

        print(f"Optimization configured with method: {method}")

        if method == "n-folds":
            print(f"  Using {folds} folds for optimization")

        print(f"Optimizing for: {objective_function}")
        print("Parameter bounds:")

        for param, (lower, upper) in bounds.items():
            print(f"    {param}: [{lower}, {upper}]")

    @staticmethod
    def _create_param_dict(keys: List[str], values: List[float]) -> Dict[str, float]:
        """
        Create a parameter dictionary from lists of keys and values.

        Parameters
        ----------
        keys : List[str]
            List of parameter names
        values : List[float]
            List of parameter values

        Returns
        -------
        Dict[str, float]
            Dictionary mapping parameter names to values
        """
        return {key: value for key, value in zip(keys, values)}

    def _objective_function(self, params: List[float]) -> float:
        """
        Calculate the objective function value for a set of parameters.

        This function is used by the optimization algorithm to evaluate
        different parameter sets.

        Parameters
        ----------
        params : List[float]
            List of parameter values in the same order as self.bounds.keys()

        Returns
        -------
        float
            Objective function value (to be minimized)
        """
        # Create a copy of the model to avoid modifying the original
        model_copy = self._model_copy.copy()

        # Create parameter dictionary from the list
        param_dict = self._create_param_dict(list(self.bounds.keys()), params)

        # Update model parameters
        model_copy.update_parameters(param_dict)

        # Run the model with the updated parameters
        results = model_copy.run(self.training_data)

        # Calculate total simulated runoff
        simulated_Q = results["Q_s"] + results["Q_gw"]

        # Get the specified metric function
        metric_func = GOF_METRICS[self.objective_function]

        # Calculate metric value
        metric_value = metric_func(simulated_Q, self.training_data["Q"])

        # Determine if we need to maximize or minimize this metric
        # For metrics where higher values are better, we negate the result
        # since optimization algorithms minimize by default
        if self.objective_function in MAXIMIZE_METRICS:
            return -metric_value
        else:
            return metric_value

    def single_fold_calibration(
        self,
        bounds_list: List[Tuple[float, float]],
        initial_guess: Optional[List[float]] = None,
        verbose: bool = False
    ) -> List[float]:
        """
        Perform a single optimization run with one initial parameter guess.

        Parameters
        ----------
        bounds_list : List[Tuple[float, float]]
            List of parameter bounds as (lower, upper) tuples
        initial_guess : List[float], optional
            Initial parameter values. If None, random values within bounds are used.
        verbose : bool, default=False
            Whether to print optimization progress information

        Returns
        -------
        List[float]
            Optimized parameter values
        """
        # Generate random initial guess if not provided
        if initial_guess is None:
            initial_guess = [
                np.random.uniform(low, high)
                for low, high in bounds_list
            ]

        if verbose:
            print(f"Starting optimization with initial guess: {initial_guess}")

        # Define callback for printing progress
        def print_progress(xk):
            if verbose:
                metric_value = self._objective_function(xk)
                if self.objective_function in MAXIMIZE_METRICS:
                    metric_value = -metric_value

                param_dict = self._create_param_dict(self.bounds.keys(), xk)
                print(f"Current parameters: {param_dict}")
                print(f"Current {self.objective_function}: {metric_value:.4f}")

        # Run optimization
        optimization_result = minimize(
            self._objective_function,
            initial_guess,
            method="L-BFGS-B",
            bounds=bounds_list,
            callback=print_progress if verbose else None,
            options={
                "ftol": 1e-6,
                "gtol": 1e-5,
                "eps": 1e-3,
                "maxiter": 100,
                "disp": verbose
            }
        )

        # Round parameter values for readability
        optimized_params = [round(param, 4) for param in optimization_result.x]

        if verbose:
            param_dict = self._create_param_dict(
                self.bounds.keys(), optimized_params)
            print(f"Optimization complete. Final parameters: {param_dict}")
            print(f"Success: {optimization_result.success}")
            if not optimization_result.success:
                print(f"Reason: {optimization_result.message}")

        return optimized_params

    def calibrate(
        self,
        initial_guess: Optional[List[float]] = None,
        verbose: bool = False
    ) -> Tuple[Dict[str, float], Optional[pd.DataFrame]]:
        """
        Calibrate the model by finding optimal parameter values.

        This method uses the optimization configuration set by `set_options`
        to find parameter values that optimize the selected objective function.

        Parameters
        ----------
        initial_guess : List[float], optional
            Initial parameter values. If None, random values within bounds are used.
        verbose : bool, default=False
            Whether to print detailed optimization progress

        Returns
        -------
        Tuple[Dict[str, float], Optional[pd.DataFrame]]
            A tuple containing:
            - Dictionary of optimal parameter values
            - DataFrame of all calibration results (for n-folds method) or None (for local method)

        Notes
        -----
        With "local" method, a single optimization is performed.
        With "n-folds" method, multiple optimizations are performed in parallel
        with different random initial guesses, and the best result is selected.
        """
        # Ensure bounds have been set
        if not self.bounds:
            raise ValueError(
                "Parameter bounds not set. Call set_options() before calibration."
            )

        # Convert parameter bounds dictionary to list of tuples for scipy.optimize
        bounds_list = list(self.bounds.values())

        # Reset calibration results
        self.calibration_results = None
        self.best_parameters = None

        print(f"Starting calibration using {self.method} method...")
        print(f"Optimizing for {self.objective_function}")

        if self.method == "local":
            # Single optimization run
            optimized_values = self.single_fold_calibration(
                bounds_list, initial_guess, verbose
            )

            # Convert to dictionary and store as best parameters
            optimized_params = self._create_param_dict(
                list(self.bounds.keys()), optimized_values
            )
            self.best_parameters = optimized_params

            # Apply best parameters to the working model
            self._model_copy.update_parameters(optimized_params)

            return optimized_params, None

        elif self.method == "n-folds":
            print(f"Running {self.folds} parallel optimizations...")

            # Run multiple optimizations in parallel
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(
                    self.single_fold_calibration,
                    [bounds_list] * self.folds,
                    [initial_guess] *
                    self.folds if initial_guess else [None] * self.folds,
                    [verbose] * self.folds
                ))

            # Convert results to DataFrame
            columns = list(self.bounds.keys())
            results_df = pd.DataFrame(results, columns=columns)
            self.calibration_results = results_df

            # Find best parameters
            best_params = self.get_best_parameters(results_df)
            self.best_parameters = best_params

            # Apply best parameters to the working model
            self._model_copy.update_parameters(best_params)

            print("Calibration complete.")
            print(f"Best parameters: {best_params}")

            return best_params, results_df

    def get_best_parameters(self, results: pd.DataFrame) -> Dict[str, float]:
        """
        Identify the best parameter set from multiple calibration results.

        Parameters
        ----------
        results : pd.DataFrame
            DataFrame containing multiple parameter sets from calibration

        Returns
        -------
        Dict[str, float]
            Dictionary of best parameter values
        """
        best_metric_value = float(
            '-inf') if self.objective_function in MAXIMIZE_METRICS else float('inf')
        best_params = None

        # Create a copy of the model for evaluation
        model_copy = self._model_copy.copy()
        metric_func = GOF_METRICS[self.objective_function]

        # Evaluate each parameter set
        for _, row in results.iterrows():
            # Get parameters from row
            params = row.to_dict()

            # Update model and run
            model_copy.update_parameters(params)
            model_results = model_copy.run(self.training_data)

            # Calculate total runoff and metric
            simulated_Q = model_results["Q_s"] + model_results["Q_gw"]
            observed_Q = self.training_data["Q"]
            current_metric = metric_func(simulated_Q, observed_Q)

            # Check if this is the best result so far
            if (self.objective_function in MAXIMIZE_METRICS and current_metric > best_metric_value) or \
                    (self.objective_function not in MAXIMIZE_METRICS and current_metric < best_metric_value):
                best_metric_value = current_metric
                best_params = params

        print(f"Best {self.objective_function} value: {best_metric_value:.4f}")
        return best_params

    def score_model(
        self,
        metrics: Optional[List[str]] = None,
        parameters: Optional[Dict[str, float]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance using various goodness-of-fit metrics.

        Parameters
        ----------
        metrics : List[str], optional
            List of metric names to calculate. If None, all available metrics are used.
        parameters : Dict[str, float], optional
            Parameter values to use for evaluation. If None, the best parameters 
            from calibration are used.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Nested dictionary with scores for training and validation data:
            {
                "training": {"metric1": value1, "metric2": value2, ...},
                "validation": {"metric1": value1, "metric2": value2, ...}
            }

        Notes
        -----
        If validation data was not provided during initialization,
        only training scores will be returned.
        """
        # Use all metrics if none specified
        if metrics is None:
            metrics = list(GOF_METRICS.keys())
        else:
            # Validate requested metrics
            invalid_metrics = set(metrics) - set(GOF_METRICS.keys())
            if invalid_metrics:
                raise ValueError(f"Invalid metrics: {invalid_metrics}. "
                                 f"Valid options are: {list(GOF_METRICS.keys())}")

        # Use best parameters if none provided
        if parameters is None:
            if self.best_parameters is None:
                raise ValueError("No parameters provided and no calibration results available. "
                                 "Run calibrate() first or provide parameters.")
            parameters = self.best_parameters

        # Create a model copy with the specified parameters
        model_copy = self._model_copy.copy()
        model_copy.update_parameters(parameters)

        # Dictionary to store results
        scores = {}

        # Evaluate on training data
        training_results = model_copy.run(self.training_data)
        simulated_Q = training_results["Q_s"] + training_results["Q_gw"]
        observed_Q = self.training_data["Q"]

        training_scores = {}
        for metric in metrics:
            metric_func = GOF_METRICS[metric]
            score = round(metric_func(simulated_Q, observed_Q), 4)
            training_scores[metric] = score

        scores["training"] = training_scores

        # Evaluate on validation data if available
        if self.validation_data is not None:
            validation_results = model_copy.run(self.validation_data)
            simulated_Q = validation_results["Q_s"] + \
                validation_results["Q_gw"]
            observed_Q = self.validation_data["Q"]

            validation_scores = {}
            for metric in metrics:
                metric_func = GOF_METRICS[metric]
                score = round(metric_func(simulated_Q, observed_Q), 4)
                validation_scores[metric] = score

            scores["validation"] = validation_scores

        # Print results in a nicely formatted table
        print("\nModel Performance Metrics:")
        print("-" * 50)

        headers = ["Metric", "Training", "Validation"] if self.validation_data is not None else [
            "Metric", "Training"]
        print(f"{headers[0]:<10}", end="")
        for h in headers[1:]:
            print(f"{h:>12}", end="")
        print()
        print("-" * 50)

        for metric in metrics:
            print(f"{metric:<10}", end="")
            print(f"{scores['training'][metric]:>12.4f}", end="")
            if self.validation_data is not None:
                print(f"{scores['validation'][metric]:>12.4f}", end="")
            print()

        return scores

    def get_simulated_results(
        self,
        dataset: str = "training",
        parameters: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Run the model and get simulation results for analysis or plotting.

        Parameters
        ----------
        dataset : str, default="training"
            Which dataset to use for simulation:
            - "training": Use training data
            - "validation": Use validation data
        parameters : Dict[str, float], optional
            Parameter values to use for simulation. If None, the best parameters
            from calibration are used.

        Returns
        -------
        pd.DataFrame
            DataFrame containing simulation results with the same index as the input data

        Raises
        ------
        ValueError
            If validation data is requested but not available,
            or if no parameters are available and none are provided
        """
        # Use best parameters if none provided
        if parameters is None:
            if self.best_parameters is None:
                raise ValueError("No parameters provided and no calibration results available. "
                                 "Run calibrate() first or provide parameters.")
            parameters = self.best_parameters

        # Choose the appropriate dataset
        if dataset.lower() == "training":
            data = self.training_data
        elif dataset.lower() == "validation":
            if self.validation_data is None:
                raise ValueError("Validation data not available.")
            data = self.validation_data
        else:
            raise ValueError(
                f"Invalid dataset: {dataset}. Use 'training' or 'validation'.")

        # Create a model copy with the specified parameters
        model_copy = self._model_copy.copy()
        model_copy.update_parameters(parameters)

        # Run the model and return results
        return model_copy.run(data)

    def get_optimized_model(self) -> BucketModel:
        """
        Get a copy of the model with optimized parameters.

        Returns
        -------
        BucketModel
            Copy of the model with optimal parameters from calibration

        Raises
        ------
        ValueError
            If no calibration has been performed
        """
        if self.best_parameters is None:
            raise ValueError(
                "No calibration results available. Run calibrate() first.")

        # Create a fresh copy of the original model
        optimized_model = self.model.copy()

        # Apply the best parameters
        optimized_model.update_parameters(self.best_parameters)

        return optimized_model

    def plot_objective_surface(
        self,
        param1: str,
        param2: str,
        n_points: int = 20,
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = "viridis",
        optimal_point: bool = True
    ) -> None:
        """
        Plot the objective function surface for two parameters.

        This method creates a contour plot showing how the objective function
        varies across the parameter space, helping visualize parameter sensitivity.

        Parameters
        ----------
        param1 : str
            Name of first parameter to vary
        param2 : str
            Name of second parameter to vary
        n_points : int, default=20
            Number of points to sample for each parameter
        figsize : Tuple[int, int], default=(10, 8)
            Figure size (width, height) in inches
        cmap : str, default="viridis"
            Colormap for contour plot
        optimal_point : bool, default=True
            Whether to mark the optimal parameter point on the plot

        Raises
        ------
        ValueError
            If parameters are not valid or calibration results are not available
        """
        # Check if parameters are valid
        if param1 not in self.bounds:
            raise ValueError(f"Parameter '{param1}' not found in bounds")
        if param2 not in self.bounds:
            raise ValueError(f"Parameter '{param2}' not found in bounds")

        # Get current best parameters (needed for holding other parameters constant)
        if self.best_parameters is None and optimal_point:
            raise ValueError(
                "No calibration results available. Run calibrate() first.")

        param_values = self.best_parameters or self._model_copy.get_parameters()

        # Create parameter grids
        param1_values = np.linspace(
            self.bounds[param1][0], self.bounds[param1][1], n_points)
        param2_values = np.linspace(
            self.bounds[param2][0], self.bounds[param2][1], n_points)
        P1, P2 = np.meshgrid(param1_values, param2_values)

        # Initialize objective function matrix
        objective_matrix = np.zeros(P1.shape)

        # Calculate objective function for each parameter combination
        for i in range(n_points):
            for j in range(n_points):
                # Create parameter dictionary with current values
                current_params = param_values.copy()
                current_params[param1] = P1[i, j]
                current_params[param2] = P2[i, j]

                # Update model and run
                self._model_copy.update_parameters(current_params)
                results = self._model_copy.run(self.training_data)
                simulated_Q = results["Q_s"] + results["Q_gw"]

                # Calculate metric
                metric_func = GOF_METRICS[self.objective_function]
                metric_value = metric_func(
                    simulated_Q, self.training_data["Q"])

                # For visualization, we want higher values to be better
                if self.objective_function not in MAXIMIZE_METRICS:
                    # Invert metrics where lower values are better
                    # We use a negative transformation to maintain the relative differences
                    objective_matrix[i, j] = -metric_value
                else:
                    objective_matrix[i, j] = metric_value

        # Create plot
        plt.figure(figsize=figsize)

        # Create contour plot
        contour = plt.contourf(P1, P2, objective_matrix, 20, cmap=cmap)
        plt.colorbar(contour, label=f"{self.objective_function} value")

        # Add contour lines for better visualization
        contour_lines = plt.contour(
            P1, P2, objective_matrix, 10, colors='black', alpha=0.5)
        plt.clabel(contour_lines, inline=True, fontsize=8, fmt='%.3f')

        # Mark optimal point if available
        if optimal_point and self.best_parameters:
            optimal_x = self.best_parameters[param1]
            optimal_y = self.best_parameters[param2]
            plt.scatter([optimal_x], [optimal_y], color='red', s=100, marker='*',
                        label='Optimal point')
            plt.legend()

        # Add labels and title
        param1_unit = self._get_parameter_unit(param1)
        param2_unit = self._get_parameter_unit(param2)

        plt.xlabel(f"{param1} [{param1_unit}]")
        plt.ylabel(f"{param2} [{param2_unit}]")
        plt.title(f"Objective function surface for {param1} and {param2}")

        plt.tight_layout()
        plt.show()

    def _get_parameter_unit(self, param: str) -> str:
        """
        Get appropriate unit for a parameter.

        Parameters
        ----------
        param : str
            Parameter name

        Returns
        -------
        str
            Unit for the parameter
        """
        units = {
            "k": "mm/°C/day",
            "S_max": "mm",
            "fr": "-",
            "rg": "days",
            "snow_threshold_temp": "°C"
        }
        return units.get(param, "-")

    def local_sensitivity_analysis(
        self,
        percent_change: float = 5.0,
        parameters: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Perform local sensitivity analysis on model parameters.

        This method analyzes how changes in each parameter affect annual runoff
        by applying a percentage change to each parameter individually.

        Parameters
        ----------
        percent_change : float, default=5.0
            Percentage change to apply to each parameter (positive number)
        parameters : List[str], optional
            List of parameters to analyze. If None, all calibrated parameters are analyzed.

        Returns
        -------
        pd.DataFrame
            DataFrame with sensitivity results for each parameter:
            - Parameter: Name of the parameter
            - Sensitivity_Pos: Sensitivity with positive parameter change
            - Sensitivity_Neg: Sensitivity with negative parameter change
            - Average_Sensitivity: Average of positive and negative sensitivities

        Notes
        -----
        Sensitivity is calculated as relative change in output divided by
        relative change in input (elasticity).
        """
        if self.best_parameters is None:
            raise ValueError(
                "No calibration results available. Run calibrate() first.")

        # Use all parameters if none specified
        if parameters is None:
            parameters = list(self.best_parameters.keys())

        # Function to compute annual runoff
        def compute_annual_runoff(model_params):
            model_copy = self._model_copy.copy()
            model_copy.update_parameters(model_params)
            results = model_copy.run(self.training_data)
            total_runoff = results["Q_s"] + results["Q_gw"]
            return total_runoff.mean()

        # Get baseline runoff with optimal parameters
        base_params = self.best_parameters.copy()
        base_runoff = compute_annual_runoff(base_params)

        sensitivity_results = []

        for param in parameters:
            param_value = base_params[param]

            # Compute runoff with positive change
            pos_params = base_params.copy()
            pos_params[param] = param_value * (1 + percent_change/100)
            pos_runoff = compute_annual_runoff(pos_params)

            # Compute runoff with negative change
            neg_params = base_params.copy()
            neg_params[param] = param_value * (1 - percent_change/100)
            neg_runoff = compute_annual_runoff(neg_params)

            # Calculate sensitivities
            # Elasticity formula: (dQ/Q)/(dP/P) = (dQ/dP)*(P/Q)
            delta_p_pos = param_value * (percent_change/100)
            delta_p_neg = -param_value * (percent_change/100)

            sens_pos = ((pos_runoff - base_runoff) / delta_p_pos) * \
                (param_value / base_runoff)
            sens_neg = ((neg_runoff - base_runoff) / delta_p_neg) * \
                (param_value / base_runoff)
            avg_sens = (abs(sens_pos) + abs(sens_neg)) / 2

            sensitivity_results.append({
                "Parameter": param,
                "Sensitivity_Pos": round(sens_pos, 4),
                "Sensitivity_Neg": round(sens_neg, 4),
                "Average_Sensitivity": round(avg_sens, 4)
            })

        # Create DataFrame and sort by average sensitivity
        sensitivity_df = pd.DataFrame(sensitivity_results)
        sensitivity_df = sensitivity_df.sort_values(
            "Average_Sensitivity", ascending=False)

        return sensitivity_df
