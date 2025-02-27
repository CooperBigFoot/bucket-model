import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime


class BucketModel:
    """
    A simple hydrological bucket model for simulating watershed processes.
    
    This model simulates snow accumulation and melt, evapotranspiration,
    surface runoff, soil moisture dynamics, and groundwater flow.
    
    Parameters
    ----------
    params : Dict[str, float]
        Dictionary of model parameters:
        - k: Degree-day snowmelt parameter [mm/°C/day]
        - S_max: Maximum soil water storage [mm]
        - fr: Fraction of impermeable area at soil saturation [-]
        - rg: Mean residence time of water in groundwater [days]
        - snow_threshold_temp: Temperature threshold for snowfall [°C]
    """
    
    def __init__(self, params: Dict[str, float]):
        # Model parameters
        self.k: float = params.get("k", 1.5)  # Degree-day factor [mm/°C/day]
        self.S_max: float = params.get("S_max", 100)  # Max soil water capacity [mm]
        self.fr: float = params.get("fr", 0.2)  # Impermeable area fraction [-]
        self.rg: float = params.get("rg", 10)  # Groundwater residence time [days]
        self.snow_threshold_temp: float = params.get("snow_threshold_temp", 0.0)  # Snow threshold [°C]
        self.snowmelt_temp_threshold: float = 0.0  # Snowmelt temperature threshold [°C]
        
        # Validate parameters
        self._validate_params()
        
        # State variables
        self.soil_moisture: float = 10.0  # Soil water content [mm]
        self.groundwater: float = 100.0  # Groundwater storage [mm]
        self.snow_cover: float = 0.0  # Snow accumulation [mm]
        
        # Temperature variables
        self.basin_temp: float = 0.0  # Basin temperature [°C]
        self.temp_max: float = 0.0  # Maximum temperature [°C]
        self.temp_min: float = 0.0  # Minimum temperature [°C]
        
        # Flux variables
        self.precip: float = 0.0  # Total precipitation [mm]
        self.rain: float = 0.0  # Rainfall [mm]
        self.snow: float = 0.0  # Snowfall [mm]
        self.snowmelt: float = 0.0  # Snowmelt [mm/day]
        self.pet: float = 0.0  # Potential evapotranspiration [mm/day]
        self.et: float = 0.0  # Actual evapotranspiration [mm/day]
        self.surface_runoff: float = 0.0  # Surface runoff [mm/day]
        self.groundwater_runoff: float = 0.0  # Groundwater runoff [mm/day]
        self.percolation: float = 0.0  # Percolation [mm/day]
        
        # Catchment properties
        self.lapse_rate: float = 0.5 / 100  # Temperature lapse rate [°C/m]
        self.basin_mean_elevation: float = 1000.0  # Basin mean elevation [m.a.s.l.]
        self.hru_mean_elevation: float = 1000.0  # HRU mean elevation [m.a.s.l.]
        self.latitude: float = 45.0  # Latitude [°]
        
        # Current simulation date
        self.date: Optional[datetime] = None

        # Initial state values for reset
        self._initial_soil_moisture: float = self.soil_moisture
        self._initial_groundwater: float = self.groundwater
        self._initial_snow_cover: float = self.snow_cover

    def _validate_params(self) -> None:
        """
        Validate model parameters to ensure they are physically reasonable.
        
        Raises
        ------
        ValueError
            If any parameter is outside its valid range
        """
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}")

        if self.S_max <= 0:
            raise ValueError(f"S_max must be positive, got {self.S_max}")

        if not 0 <= self.fr <= 1:
            raise ValueError(f"fr must be between 0 and 1, got {self.fr}")

        if self.rg < 1:
            raise ValueError(f"rg must be greater than 1, got {self.rg}")

    def reset_state(self) -> None:
        """
        Reset model state variables to their initial values.
        
        This is useful when running multiple simulations with the same model instance.
        """
        self.soil_moisture = self._initial_soil_moisture
        self.groundwater = self._initial_groundwater
        self.snow_cover = self._initial_snow_cover
        
        # Reset flux variables
        self.precip = 0.0
        self.rain = 0.0
        self.snow = 0.0
        self.snowmelt = 0.0
        self.pet = 0.0
        self.et = 0.0
        self.surface_runoff = 0.0
        self.groundwater_runoff = 0.0
        self.percolation = 0.0
        
        # Reset temperature variables
        self.basin_temp = 0.0
        self.temp_max = 0.0
        self.temp_min = 0.0

    def set_catchment_properties(self, properties: Dict[str, float]) -> None:
        """
        Set catchment properties for the model.
        
        Parameters
        ----------
        properties : Dict[str, float]
            Dictionary containing catchment properties:
            - lapse_rate: Temperature lapse rate [°C/m]
            - basin_mean_elevation: Basin mean elevation [m.a.s.l.]
            - hru_mean_elevation: HRU mean elevation [m.a.s.l.]
            - snowmelt_temp_threshold: Temperature threshold for snowmelt [°C]
            - latitude: Latitude of the catchment [°]
        """
        if "lapse_rate" in properties:
            self.lapse_rate = properties["lapse_rate"]
        if "basin_mean_elevation" in properties:
            self.basin_mean_elevation = properties["basin_mean_elevation"]
        if "hru_mean_elevation" in properties:
            self.hru_mean_elevation = properties["hru_mean_elevation"]
        if "snowmelt_temp_threshold" in properties:
            self.snowmelt_temp_threshold = properties["snowmelt_temp_threshold"]
        if "latitude" in properties:
            self.latitude = properties["latitude"]
        
        print("Catchment properties set:")
        print(f"  Lapse rate: {self.lapse_rate} °C/m")
        print(f"  Basin mean elevation: {self.basin_mean_elevation} m.a.s.l.")
        print(f"  HRU mean elevation: {self.hru_mean_elevation} m.a.s.l.")
        print(f"  Snowmelt temperature threshold: {self.snowmelt_temp_threshold} °C")
        print(f"  Latitude: {self.latitude}°")

    def change_initial_conditions(self, soil_moisture: Optional[float] = None, 
                                 groundwater: Optional[float] = None,
                                 snow_cover: Optional[float] = None) -> None:
        """
        Change the initial conditions of the model.
        
        Parameters
        ----------
        soil_moisture : float, optional
            New initial soil moisture [mm]
        groundwater : float, optional
            New initial groundwater storage [mm]
        snow_cover : float, optional
            New initial snow cover [mm]
            
        Raises
        ------
        ValueError
            If any value is outside its valid range
        """
        if soil_moisture is not None:
            if 0 <= soil_moisture <= self.S_max:
                self.soil_moisture = soil_moisture
                self._initial_soil_moisture = soil_moisture
            else:
                raise ValueError(f"Soil moisture must be between 0 and {self.S_max} mm")
        
        if groundwater is not None:
            if groundwater >= 0:
                self.groundwater = groundwater
                self._initial_groundwater = groundwater
            else:
                raise ValueError("Groundwater storage must be non-negative")
        
        if snow_cover is not None:
            if snow_cover >= 0:
                self.snow_cover = snow_cover
                self._initial_snow_cover = snow_cover
            else:
                raise ValueError("Snow cover must be non-negative")
        
        print("Initial conditions updated:")
        print(f"  Soil moisture: {self.soil_moisture} mm")
        print(f"  Groundwater storage: {self.groundwater} mm")
        print(f"  Snow cover: {self.snow_cover} mm")

    def adjust_temperature(self) -> None:
        """
        Adjust temperature based on lapse rate and elevation difference.
        
        This method calculates the basin temperature based on the station temperature
        and the elevation difference between the basin and the station.
        """
        T_mean = (self.temp_max + self.temp_min) / 2
        delta_h = self.basin_mean_elevation - self.hru_mean_elevation
        lr_adjustment = self.lapse_rate * delta_h
        
        self.basin_temp = T_mean + lr_adjustment
        self.temp_max += lr_adjustment
        self.temp_min += lr_adjustment

    def partition_precipitation(self) -> None:
        """
        Partition precipitation into rainfall and snowfall.
        
        The partitioning is based on the daily air temperature following
        the procedure described in the PRMS model.
        """
        if self.temp_min > self.snow_threshold_temp:
            # All precipitation is rain when min temp is above threshold
            self.rain = self.precip
            self.snow = 0.0
        elif self.temp_max <= self.snow_threshold_temp:
            # All precipitation is snow when max temp is below threshold
            self.snow = self.precip
            self.rain = 0.0
        else:
            # Mixed precipitation based on temperature range
            # Assumes linear distribution of temperature between min and max
            rain_fraction = self.temp_max / (self.temp_max - self.temp_min)
            self.rain = self.precip * rain_fraction
            self.snow = self.precip - self.rain

    def compute_snow_melt(self) -> None:
        """
        Compute snowmelt using the degree-day method.
        
        Snowmelt occurs when the basin temperature exceeds the
        snowmelt temperature threshold, at a rate determined by
        the degree-day factor k.
        """
        if self.basin_temp <= self.snowmelt_temp_threshold:
            # No snowmelt when temperature is below threshold
            self.snowmelt = 0.0
        else:
            # Snowmelt based on temperature excess and available snow cover
            potential_melt = self.k * (self.basin_temp - self.snowmelt_temp_threshold)
            self.snowmelt = min(potential_melt, self.snow_cover)

    def update_snow_cover(self) -> None:
        """
        Update snow cover based on snowfall and snowmelt.
        
        Snow cover increases with snowfall and decreases with snowmelt.
        """
        self.snow_cover += self.snow - self.snowmelt
        # Ensure snow cover doesn't go negative
        self.snow_cover = max(self.snow_cover, 0.0)

    def compute_julian_day(self) -> int:
        """
        Compute the Julian day (day of year) from the current date.
        
        Returns
        -------
        int
            Julian day (1-366)
        """
        return self.date.timetuple().tm_yday

    def compute_evapotranspiration(self) -> None:
        """
        Compute evapotranspiration using the Hamon method.
        
        This method calculates potential evapotranspiration based on
        day length and temperature, and then adjusts it to actual
        evapotranspiration based on available soil moisture.
        """
        # Calculate Julian day
        julian_day = self.compute_julian_day()
        
        # Calculate declination angle
        phi = np.radians(self.latitude)
        delta = 0.4093 * np.sin((2 * np.pi / 365) * julian_day - 1.405)
        
        # Calculate day length
        omega_s = np.arccos(-np.tan(phi) * np.tan(delta))
        day_length = 24 * omega_s / np.pi
        
        # Calculate saturated vapor pressure
        a, b, c = 0.6108, 17.27, 237.3
        es = a * np.exp(b * self.basin_temp / (self.basin_temp + c))
        
        # Calculate potential evapotranspiration using Hamon method
        self.pet = 2.1 * (day_length**2) * es / (self.basin_temp + 273.3)
        
        # Calculate actual evapotranspiration based on soil moisture availability
        rel_soil_moisture = self.soil_moisture / self.S_max
        self.et = self.pet * rel_soil_moisture

    def update_soil_moisture(self) -> None:
        """
        Update soil moisture accounting for all water fluxes.
        
        This method implements the water dynamics in the soil bucket,
        computing surface runoff and percolation based on soil moisture content.
        """
        # Calculate water input to soil
        water_input = self.rain + self.snowmelt
        
        # Calculate potential soil water content
        potential_soil_water = self.soil_moisture + water_input - self.et
        
        # Surface runoff generation when soil is saturated
        if potential_soil_water > self.S_max:
            self.surface_runoff = water_input * self.fr
            potential_soil_water -= self.surface_runoff
            
            # Percolation to groundwater when soil is still saturated
            if potential_soil_water > self.S_max:
                self.percolation = potential_soil_water - self.S_max
                self.soil_moisture = self.S_max
            else:
                self.soil_moisture = potential_soil_water
                self.percolation = 0.0
        else:
            # No runoff or percolation when soil is not saturated
            self.soil_moisture = max(potential_soil_water, 0.0)
            self.surface_runoff = 0.0
            self.percolation = 0.0

    def update_groundwater(self) -> None:
        """
        Update groundwater storage and compute baseflow.
        
        This method applies the linear reservoir concept to compute
        groundwater runoff and update groundwater storage.
        """
        # Linear reservoir outflow
        self.groundwater_runoff = self.groundwater / self.rg
        
        # Update groundwater storage
        self.groundwater += self.percolation - self.groundwater_runoff
        
        # Ensure groundwater storage doesn't go negative
        self.groundwater = max(self.groundwater, 0.0)

    def step(self, date: datetime, precip: float, tmax: float, tmin: float) -> Dict[str, float]:
        """
        Perform one time step of the model.
        
        Parameters
        ----------
        date : datetime
            Current simulation date
        precip : float
            Total precipitation [mm]
        tmax : float
            Maximum temperature [°C]
        tmin : float
            Minimum temperature [°C]
            
        Returns
        -------
        Dict[str, float]
            Model outputs for the time step including:
            - ET: Actual evapotranspiration [mm/day]
            - Q_s: Surface runoff [mm/day]
            - Q_gw: Groundwater runoff [mm/day]
            - Total_Runoff: Total runoff (Q_s + Q_gw) [mm/day]
            - Snow_accum: Snow accumulation [mm]
            - S: Soil water content [mm]
            - S_gw: Groundwater storage [mm]
            - Snow_melt: Snowmelt [mm/day]
            - Rain: Rainfall [mm]
            - Snow: Snowfall [mm]
            - Precip: Total precipitation [mm]
        """
        # Set current time step inputs
        self.date = date
        self.precip = precip
        self.temp_max = tmax
        self.temp_min = tmin
        
        # Run all hydrological processes
        self.adjust_temperature()
        self.partition_precipitation()
        self.compute_snow_melt()
        self.update_snow_cover()
        self.compute_evapotranspiration()
        self.update_soil_moisture()
        self.update_groundwater()
        
        # Return results for this time step
        return {
            "ET": self.et,
            "Q_s": self.surface_runoff,
            "Q_gw": self.groundwater_runoff,
            "Total_Runoff": self.surface_runoff + self.groundwater_runoff,
            "Snow_accum": self.snow_cover,
            "S": self.soil_moisture,
            "S_gw": self.groundwater,
            "Snow_melt": self.snowmelt,
            "Rain": self.rain,
            "Snow": self.snow,
            "Precip": self.precip,
        }

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run the model for a period of time.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with columns ['P_mix', 'T_max', 'T_min'] and datetime index
            
        Returns
        -------
        pd.DataFrame
            Model results with columns corresponding to model outputs
            and the same index as the input data
        """
        # Reset model state before running
        self.reset_state()
        results = []
        
        # Run model for each time step
        for date, row in data.iterrows():
            step_result = self.step(
                date, row["P_mix"], row["T_max"], row["T_min"])
            results.append(step_result)
        
        # Compile results into DataFrame
        return pd.DataFrame(results, index=data.index)

    def get_parameters(self) -> Dict[str, float]:
        """
        Return current parameter values.
        
        Returns
        -------
        Dict[str, float]
            Dictionary of current model parameters
        """
        return {
            "k": self.k,
            "S_max": self.S_max,
            "fr": self.fr,
            "rg": self.rg,
            "snow_threshold_temp": self.snow_threshold_temp
        }

    def update_parameters(self, new_params: Dict[str, float], verbose: bool = False) -> None:
        """
        Update model parameters.
        
        Parameters
        ----------
        new_params : Dict[str, float]
            Dictionary of parameters to update
        verbose : bool, optional
            If True, print updated parameters
            
        Raises
        ------
        ValueError
            If any parameter is outside its valid range
        """
        # Update parameters
        if "k" in new_params:
            self.k = new_params["k"]
        if "S_max" in new_params:
            self.S_max = new_params["S_max"]
        if "fr" in new_params:
            self.fr = new_params["fr"]
        if "rg" in new_params:
            self.rg = new_params["rg"]
        if "snow_threshold_temp" in new_params:
            self.snow_threshold_temp = new_params["snow_threshold_temp"]
        
        # Validate updated parameters
        self._validate_params()

        # Print updated parameters if verbose
        if verbose:    
            print("Parameters updated:")
            print(f"  k: {self.k} mm/°C/day")
            print(f"  S_max: {self.S_max} mm")
            print(f"  fr: {self.fr}")
            print(f"  rg: {self.rg} days")
            print(f"  snow_threshold_temp: {self.snow_threshold_temp} °C")

    def copy(self) -> 'BucketModel':
        """
        Create a deep copy of the model.
        
        Returns
        -------
        BucketModel
            A new BucketModel instance with the same parameters and state
        """
        import copy
        return copy.deepcopy(self)