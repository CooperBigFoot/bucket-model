import numpy as np
import pandas as pd


class BucketModel:
    def __init__(self, max_soil_water_storage,  daily_degree_day_snowmelt, rain_to_runoff, ground_water_residence_time, infiltration_rate = 0, mean_elevation = 2035):
        # Define the initial values of buckets
        self.S = 10  # Soil water storage
        self.S_gw = 100  # Ground water storage

        # Define model attributes
        self.P = None  # Precipitation
        self.T_max = 0
        self.T_min = 0
        self.T_basin = 0
        self.Snow = 0
        self.Melting_snow = 0
        self.Snow_cover = 0
        self.Rain = 0
        self.ET = 0
        self.Percolation = 0
        self.Q_s = 0
        self.Q_gw = 0
        self.Date = 0
        self.Q_inf = 0
        self.Q_sat = 0
        self.T_sm = 0  # Snowmelt threshold temperature

        # Define model parameters
        self.S_max = max_soil_water_storage
        self.K = daily_degree_day_snowmelt  # Snowmelt rate
        self.fr = rain_to_runoff  # Fraction of rainfall that becomes surface runoff
        self.Rg = ground_water_residence_time  # Residency time of groundwater
        self.Imax = infiltration_rate # Maximum infiltration rate in unsaturated zone

        # Define model constants
        self.H_station = 1638
        self.H_basin = mean_elevation
        self.lr = 0.5/100  # Lapse rate

    def compute_julianday(self):
        """
        Compute the Julian day based on self.Date.

        Returns:
        - J: Julian day (1 for January 1, ... , 365 or 366 for December 31).
        """
        # Calculate and return the Julian day
        J = self.Date.timetuple().tm_yday
        return J

    def calculate_evapotranspiration(self, lat=46.9):
        """
        Calculate daily evapotranspiration.

        Parameters:
        - lat: Latitude of the catchment in degrees.
        """
        J = self.compute_julianday()

        # Convert latitude from degrees to radians
        phi = np.radians(lat)

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

        # If temperature is below or equal to zero, evapotranspiration is zero
        if self.T_basin <= 0:
            self.ET = 0

        # If potential evapotranspiration is negative, set it to zero
        elif E < 0:
            E = 0
            self.ET = self.S / self.S_max * E

        # Actual evapotranspiration
        else:
            self.ET = self.S / self.S_max * E

    def compute_basin_temperature(self):
        """Compute average basin temperature.

        The basin temperature is calculated as the mean of maximum and minimum temperatures, adjusted for elevation differences.
        """
        T = (self.T_max + self.T_min) / 2
        self.T_basin = T + (self.H_station - self.H_basin) * self.lr

        self.T_max = self.T_max + (self.H_station - self.H_basin) * self.lr
        self.T_min = self.T_min + (self.H_station - self.H_basin) * self.lr

    def partition_precipitation(self):
        """Partition precipitation into rainfall (R) and snowfall (S) based on temperature thresholds.

        Parameters:
        - p: daily precipitation value.

        Process:
        - If minimum temperature is above freezing, all precipitation is rainfall.
        - If maximum temperature is below freezing, all precipitation is snowfall.
        - Otherwise, partition based on temperature range.
        """
        if self.T_min > 0:
            self.Rain = self.P
            self.Snow = 0
        elif self.T_max <= 0:
            self.Snow = self.P
            self.Rain = 0
        else:
            self.Rain = self.P * ((self.T_max) / (self.T_max - self.T_min))
            self.Snow = self.P - self.Rain

    def update_soil_water_storage(self):
        """Update soil water storage based on rainfall, snowmelt, and evapotranspiration."""

        if self.Imax:
            self.update_soil_water_storage_upgraded()

        else:
            potential_soil_water_content = self.S + self.Rain + self.Melting_snow - self.ET

            if potential_soil_water_content > self.S_max:
                self.surface_runoff()

            potential_soil_water_content -= self.Q_s

            if potential_soil_water_content > self.S_max:
                self.S = self.S_max
                self.percolation(potential_soil_water_content - self.S_max)

            else:
                self.S = max(potential_soil_water_content, 0)

    def update_soil_water_storage_upgraded(self):

        potential_soil_water_content = self.S + self.Rain + self.Melting_snow - self.ET
        potential_soil_infiltration = self.Rain + self.Melting_snow

        self.Q_inf = 0
        if potential_soil_infiltration > self.Imax:
            self.Q_inf = potential_soil_infiltration - self.Imax # Infiltration excess runoff

        potential_soil_water_content -= self.Q_inf

        # Now back to the standard model:

        self.Q_sat = 0
        if potential_soil_water_content > self.S_max:
            self.Q_sat += self.fr*(self.Rain + self.Melting_snow - self.Q_inf) # self.fr*(self.Imax)

        potential_soil_water_content -= self.Q_sat

        if potential_soil_water_content > self.S_max:
            self.S = self.S_max
            self.percolation(potential_soil_water_content - self.S_max)

        else:
            self.S = max(potential_soil_water_content, 0)

        self.Q_s = self.Q_inf + self.Q_sat

    def surface_runoff(self):
        self.Q_s = (self.Rain + self.Melting_snow) * self.fr

    def percolation(self, excess):
        self.Percolation = excess

    def groundwater_runoff(self):
        self.Q_gw = self.S_gw / self.Rg

    def reset_values(self):
        self.Q_s = 0
        self.Q_gw = 0
        self.Percolation = 0
        self.T_basin = 0
        self.Rain = 0
        self.Snow = 0
        self.ET = 0
        self.Melting_snow = 0
        self.P = 0
        self.Q_inf = 0
        self.Q_sat = 0


    def update_groundwater_storage(self):
        self.groundwater_runoff()
        self.S_gw += self.Percolation - self.Q_gw

        if self.S_gw < 0:
            self.S_gw = 0

    def compute_snow_melting(self):
        """Compute snowmelt based on basin temperature.

        Snow melts when the basin temperature is above freezing. The melt rate is determined by a function of basin temperature, up to a maximum of the current snow cover.
        """
        if self.T_basin > self.T_sm:
            self.Melting_snow = min(
                self.K*(self.T_basin - self.T_sm), self.Snow_cover)
        else:
            self.Melting_snow = 0

        self.update_snow_cover()

    def update_snow_cover(self):
        """Update snow cover based on snowfall and snowmelt."""
        self.Snow_cover += self.Snow - self.Melting_snow

    def run(self, data):
        """
        Execute the model for a given time series of meteorological data.

        Parameters:
        - data: A DataFrame containing the necessary input data (e.g., date, precipitation, T_max, T_min).
        """

        # Prepare arrays to store results
        ET_results = []
        Qs_results = []
        Qgw_results = []
        Snow_cover_results = []
        S_results = []
        S_gw_results = []
        Snow_melt_results = []
        Rain_results = []
        Snow_results = []
        Precipitation_results = []
        Q_inf_results =[]
        Q_sat_results = []
     

        for index, row in data.iterrows():
            # Reset values
            self.reset_values()

            self.Date = index
            self.P = row['P_mix'] # * 1.05 # 5% increase in precipitation due to undercatch
            self.T_max = row['T_max']
            self.T_min = row['T_min']

            # Compute basin temperature
            self.compute_basin_temperature()

            # Partition precipitation into rainfall and snowfall
            self.partition_precipitation()

            # Compute snow melting
            self.compute_snow_melting()

            # Calculate daily evapotranspiration
            self.calculate_evapotranspiration()

            # Update soil water storage
            self.update_soil_water_storage()
        
            # self.update_soil_water_storage()

            # Update groundwater storage
            self.update_groundwater_storage()

            # Store results for this timestep
            ET_results.append(self.ET)
            Qs_results.append(self.Q_s)
            Qgw_results.append(self.Q_gw)
            Snow_cover_results.append(self.Snow_cover)
            S_results.append(self.S)
            S_gw_results.append(self.S_gw)
            Snow_melt_results.append(self.Melting_snow)
            Rain_results.append(self.Rain)
            Snow_results.append(self.Snow)
            Precipitation_results.append(self.P)
            Q_inf_results.append(self.Q_inf)
            Q_sat_results.append(self.Q_sat)
         


        # Convert results to a DataFrame for easier handling and return
        results = pd.DataFrame({
            'ET': ET_results,
            'Qs': Qs_results,
            'Qgw': Qgw_results,
            'Snow_cover': Snow_cover_results,
            'S': S_results,
            'S_gw': S_gw_results,
            'Snow_melt': Snow_melt_results,
            'Rain': Rain_results,
            'Snow': Snow_results,
            'Precipitation': Precipitation_results,
            'Q_inf': Q_inf_results,
            'Q_sat': Q_sat_results
        }, index=data.index)

        return results