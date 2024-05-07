import pandas as pd
import numpy as np
from dataclasses import dataclass, field 
import copy

@dataclass
class BucketModel:
    k : float # degree-day snowmelt parameter
    S_max : float # max soil water storage
    fr : float # fraction of impermeable area at soil saturation 
    rg : float # mean residence time of water in grondwater
    gauge_adj : float # parameter to adjust for undercatch by rain gauge. Fractional value

    S : float = field(default=10, init=False, repr=False) # soil water content. Initial condition can be changed
    S_gw : float = field(default = 100, init=False, repr=False) # groundwater storage. Initial condition can be changed
    T_basin : float = field(default=0, init=False, repr=False) # basin temperature
    T_max : float = field(default=0, init=False, repr=False) # max temperature
    T_min : float = field(default=0, init=False, repr=False) # min temperature
    Precip : float = field(default=0, init=False, repr=False) # precipitation
    Rain : float = field(default=0, init=False, repr=False) # rainfall
    Snow : float = field(default=0, init=False, repr=False) # snow
    Snow_accum : float = field(default=0, init=False, repr=False) # snow accumulation (cover)
    Snow_melt : float = field(default=0, init=False, repr=False) # snow melt
    PET : float = field(default=0, init=False, repr=False) # potential evapotranspiration
    ET : float = field(default=0, init=False, repr=False) # evapotranspiration
    Q_s : float = field(default=0, init=False, repr=False) # surface runoff
    Q_gw : float = field(default=0, init=False, repr=False) # groundwater runoff
    Percol : float = field(default=0, init=False, repr=False) # percolation
    Date : pd.Timestamp = field(default=pd.Timestamp('2000-08-14'), init=False, repr=False) # date (I set my birthday as default date for fun :) )


    LR : float = field(init=False, repr=False) # lapse rate
    H_STATION : float = field(init=False, repr=False) # station elevation
    H_BASIN : float = field(init=False, repr=False) # basin elevation
    T_SM : float = field(init=False, repr=False) # snowmelt temperature threshold
    LAT : float = field(init=False, repr=False) # latitude in degrees

    def __post_init__(self):
        """Checking that the initialised values make sense. This prevents unrelistic results further on."""
        
        if self.k <= 0:
            raise ValueError("k must be positive")
        if self.S_max <= 0:
            raise ValueError("S_max must be positive")
        if self.fr < 0 or self.fr > 1:
            raise ValueError("fr must be between 0 and 1")
        if self.rg < 1: # Think why that is
            raise ValueError("rg must be greater than 1")
        
    def set_catchment_properties(self, lapse_rate: float, station_elevation: float, basin_elevation: float, snowmelt_temp_threshold: float, latitude: float) -> None:
        """Set the values of the constants.

        Parameters:
        - lapse_rate: lapse rate
        - station_elevation: station elevation
        - basin_elevation: basin elevation
        - snowmelt_temp_threshold: snowmelt temperature threshold
        - latitude: latitude in degrees

        Returns:
        - None
        """
        self.LR = lapse_rate # °C/m
        self.H_STATION = station_elevation # m.a.s.l
        self.H_BASIN = basin_elevation # m.a.s.l
        self.T_SM = snowmelt_temp_threshold # °C
        self.LAT = latitude # °N

    # TODO: implement change_initial_conditions method
    def change_initial_conditions(self) -> None:
        """
        This method changes the initial conditions of the model.
        """
        raise NotImplementedError("This method is not implemented yet.")
    

    def adjust_temperature(self) -> None:
        # Compute 'mean' basin temperature
        T = (self.T_max + self.T_min) / 2

        DELTA_H = self.H_STATION - self.H_BASIN

        # Adjust temperature
        LR_DELTA_H = self.LR * DELTA_H
        self.T_basin = T + LR_DELTA_H
        self.T_max += LR_DELTA_H
        self.T_min += LR_DELTA_H

    def gauge_adjustment(self) -> None:
        """Adjust for undercatch by the rain gauge."""
        self.Precip = self.Precip * (1 + self.gauge_adj)

    def partition_precipitation(self) -> None:
        """Partition precipitation into rainfall and snowfall based on temperature thresholds.

        Process:
        - If minimum temperature is above freezing, all precipitation is rainfall.
        - If maximum temperature is below freezing, all precipitation is snowfall.
        - Otherwise, partition based on temperature range.
        """
        if self.T_min > 0:
            self.Rain = self.Precip
            self.Snow = 0
        elif self.T_max <= 0:
            self.Snow = self.Precip
            self.Rain = 0
        else:
            rain_fraction = self.T_max / (self.T_max - self.T_min)
            self.Rain = self.Precip * rain_fraction
            self.Snow = self.Precip - self.Rain

    def compute_snow_melt(self) -> None:
        """Compute snowmelt based on basin temperature.

        Process:
        - If basin temperature is below snowmelt threshold, no snowmelt.
        - If basin temperature is above snowmelt threshold, melt as long as there is snow cover.
        """
        if self.T_basin <= self.T_SM:
            self.Snow_melt = 0
        else:
            self.Snow_melt = min(
                self.k * (self.T_basin - self.T_SM), self.Snow_accum
            )

    def update_snow_accum(self) -> None:
        """Update snow cover based on snowfall and snowmelt."""
        self.Snow_accum += self.Snow - self.Snow_melt
    
    def compute_julianday(self) -> int:
        """Compute the Julian day based on self.Date.

        Returns:
        - J: Julian day (1 for January 1, ... , 365 or 366 for December 31).
        """
        J = self.Date.timetuple().tm_yday
        return J
    
    def _Hamon_PET(self) -> float:
        """Calculate potential evapotranspiration using Hamon (1961): https://ascelibrary.org/doi/10.1061/JYCEAJ.0000599.

        Important:
        - the '_' prefix indicates that this method is intended for internal use only.
        """
        J = self.compute_julianday()

        # Convert latitude from degrees to radians
        phi = np.radians(self.LAT)

        # Solar declination angle
        delta = 0.4093 * np.sin((2 * np.pi / 365) * J - 1.405)

        # Sunset hour angle
        omega_s = np.arccos(-np.tan(phi) * np.tan(delta))

        # Maximum possible sunshine duration
        Nt = 24 * omega_s / np.pi

        # Saturated vapor pressure (using the temperature of the basin)
        a, b, c = 0.6108, 17.27, 237.3
        es = a * np.exp(b * self.T_basin / (self.T_basin + c))

        # Potential evapotranspiration
        E = (2.1 * (Nt ** 2) * es / (self.T_basin + 273.3))

        return E
    
    def compute_evapotranspiration(self) -> None:
        """Compute evapotranspiration using the Hamon method.

        Parameters:
        - lat: Latitude of the catchment in degrees.
        """
        self.PET = self._Hamon_PET()

        rel_soil_moisture = self.S / self.S_max
        self.ET = self.PET * rel_soil_moisture

    def surface_runoff(self) -> None:
        self.Q_s = (self.Rain + self.Snow_melt) * self.fr

    def percolation(self, excess_water) -> None:
        self.Percol = excess_water

    def update_soil_moisture(self) -> None:
        """This function implements the water dynamics in the soil bucket.
        
        Process:
        - Compute potential soil water content.
        - If potential soil water content exceeds the maximum storage, compute surface runoff.
        - Subtract surface runoff from potential soil water content.
        - If potential soil water content still exceeds the maximum storage, compute percolation.
        - Update soil water content.
        """ 
        potential_soil_water_content = self.S + self.Rain + self.Snow_melt - self.ET

        if potential_soil_water_content > self.S_max:
            self.surface_runoff()

        potential_soil_water_content -= self.Q_s

        if potential_soil_water_content > self.S_max:
            self.S = self.S_max

            # the rest is percolation
            water_excess = potential_soil_water_content - self.S_max
            self.percolation(water_excess)

        else:
            self.S = max(
                potential_soil_water_content, 0
            )

    def groundwater_runoff(self) -> None:
        """Compute groundwater runoff with the linear reservoir concept.
        
        Bonus quiz for exam preparation :) : What is the minimal value of rg?
        """
        self.Q_gw = self.S_gw / self.rg

    def update_groundwater_storage(self) -> None:
        """Update groundwater storage based on groundwater runoff."""
        self.S_gw += self.Percol - self.Q_gw

        if self.S_gw < 0:
            self.S_gw = 0

    def reset_variables(self) -> None:
        """Reset the state variables to their initial values."""
        self.Precip = 0
        self.Rain = 0
        self.Snow = 0
        self.Snow_melt = 0
        self.PET = 0
        self.ET = 0
        self.Q_s = 0
        self.Q_gw = 0
        self.Percol = 0

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the model.

        Parameters:
        - data: DataFrame with columns 'date', 'P_mix', 'T_max', 'T_min'.
        """
    
        intermediate_results = {
            'ET': [],
            'Q_s': [],
            'Q_gw': [],
            'Snow_accum': [],
            'S': [],
            'S_gw': [],
            'Snow_melt': [],
            'Rain': [],
            'Snow': [],
            'Precip': []
        }

        for index, row in data.iterrows():
            self.reset_variables()

            self.Date = index
            self.Precip = row['P_mix']  
            self.T_max = row['T_max']
            self.T_min = row['T_min']

            # Model execution steps
            self.gauge_adjustment()
            self.adjust_temperature()
            self.partition_precipitation()
            self.compute_snow_melt()
            self.update_snow_accum()
            self.compute_evapotranspiration()
            self.update_soil_moisture()
            self.groundwater_runoff()
            self.update_groundwater_storage()

            # Collect intermediate results
            for key in intermediate_results:
                intermediate_results[key].append(getattr(self, key))

        results_df = pd.DataFrame(intermediate_results, index=data.index)

        # Update column names to include the units
        return results_df
    
    def update_parameters(self, parameters: dict) -> None:
        """This function updates the model parameters.
        
        Parameters:
        - parameters (dict): A dictionary containing the parameters to update."""
        
        for key, value in parameters.items():
            setattr(self, key, value)

    def get_parameters(self) -> dict:
        """This function returns the model parameters."""
        return self.__dict__
    
    def copy(self) -> 'BucketModel':
        """This function returns a copy of the model."""
        return copy.deepcopy(self)

        



