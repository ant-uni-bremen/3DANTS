import numpy as np

class Atmospheric_loss:
    """
    A class to calculate atmospheric attenuation for satellite communications.
    
    This class implements the ITU models for atmospheric attenuation including:
    - Cloud attenuation (ITU-R P.840-3)
    - Rain attenuation
    - Gas attenuation (oxygen and water vapor)
    - Tropospheric scintillation
    - Ionospheric scintillation (for frequencies < 3 GHz)
    """
    
    def __init__(self, f_c, elevation_angle, **kwargs):
        """
        Initialize the Atmospheric_loss class with required parameters.
        
        Parameters:
        -----------
        f_c : float
            Carrier frequency in GHz
        elevation_angle : float
            Elevation angle in degrees
            
        Optional Parameters:
        -------------------
        LWC : float
            Liquid water content in kg/m^2 (default: 0.41)
        T : float
            Temperature in Kelvin (default: 273)
        hs : float
            Height of the transmitter above sea level in kilometers (default: 0.563)
        Lat : float
            Latitude in degrees (default: 47)
        r : float
            Rain rate in mm/hr (default: 40)
        p : float
            Pressure in hPa (default: 1020)
        RH : float
            Relative humidity in percentage (default: 50)
        D : float
            Antenna diameter in meters (default: 3.6)
        antenna_efficiency : float
            Antenna efficiency (default: 0.5)
        """
        # Required parameters
        self.f = f_c  # Frequency in GHz
        self.Elev = elevation_angle  # Elevation angle in degrees
        
        # Optional parameters with default values
        self.LWC = kwargs.get('LWC', 0.41)  # Liquid water content in kg/m^2
        self.T = kwargs.get('T', 273)  # Temperature in Kelvin
        self.hs = kwargs.get('hs', 0.563)  # Height of the transmitter above sea level in kilometers
        self.Lat = kwargs.get('Lat', 47)  # Latitude in degrees
        self.r = kwargs.get('r', 40)  # Rain rate in mm/hr
        self.p = kwargs.get('p', 1020)  # Pressure in hPa
        self.RH = kwargs.get('RH', 50)  # Relative humidity in percentage
        self.D = kwargs.get('D', 3.6)  # Antenna diameter in meters
        self.antenna_efficiency = kwargs.get('antenna_efficiency', 0.5)  # Antenna efficiency
        
        # Other derived parameters
        self.T_celcius = self.T - 273  # Temperature in Celsius
        
    def calculate_cloud_attenuation(self):
        """Calculate cloud attenuation based on ITU-R P.840-3"""
        # Step 1: Calculation of the principal & secondary relaxation frequencies
        theta = 300 / self.T  # constant
        fp = 20.09 - 142 * (theta - 1) + 294 * (theta - 1)**2  # Primary relaxation frequency (GHz)
        fs = 590 - 1500 * (theta - 1)  # Secondary relaxation frequency (GHz)
        
        # Step 2: Calculation of the complex dielectric permittivity of water
        eo = 77.6 + 103.3 * (theta - 1)  # constant
        e1 = 5.48  # constant
        e2 = 3.51  # constant
        eta2 = (self.f / fp) * ((eo - e1) / (1 + (self.f / fp)**2)) + (self.f / fs) * ((e1 - e2) / (1 + (self.f / fs)**2))
        eta1 = ((eo - e1) / (1 + (self.f / fp)**2)) + ((e1 - e2) / (1 + (self.f / fs)**2)) + e2  # Complex dielectric permittivity of water
        n = (2 + eta1) / eta2
        
        # Step 3: cloud specific attenuation coefficient ((dB/km)/(g/m^3))
        Kl = 0.819 * self.f / (eta2 * (1 + n**2))  # Specific Cloud attenuation coefficient((dB/km)/(g/m^3))
        
        # Step 4: cloud attenuation
        CAtten = self.LWC * Kl / np.sin(np.radians(self.Elev))  # cloud attenuation in dB
        
        return abs(CAtten)
    
    def calculate_rain_attenuation(self):
        """Calculate rain attenuation"""
        # Rain height calculation
        hr = 5 - 0.075 * (self.Lat - 23) if self.Lat > 23 else 5  # rain height for latitude in km
        x = np.radians(self.Elev)  # convert elevation angle to radians
        ls = (hr - self.hs) / np.sin(x)  # slant path length
        lg = ls * np.cos(x)  # horizontal projection length in km
        
        # Calculate kH (frequency-dependent parameter)
        log_k = 0
        a = [-5.33980, -0.35351, -0.23789, -0.94158]
        b = [-0.10008, 1.26970, 0.86036, 0.64552]
        c = [1.13098, 0.45400, 0.15354, 0.16817]
        m = -0.18961
        d = 0.71147
        
        for j in range(4):
            log_k += a[j] * np.exp(-((np.log10(self.f) - b[j]) / c[j])**2)
        
        log_k += (m * np.log10(self.f)) + d
        kH = 10**log_k
        
        # Calculate kV (frequency-dependent parameter)
        log_k1 = 0
        a1 = [-3.80595, -3.44965, -0.39902, 0.50167]
        b1 = [0.56934, -0.22911, 0.73042, 1.07319]
        c1 = [0.81061, 0.51059, 0.11899, 0.27195]
        x1 = -0.16398
        d1 = 0.63297
        
        for l in range(4):
            log_k1 += a1[l] * np.exp(-((np.log10(self.f) - b1[l]) / c1[l])**2)
        
        log_k1 += (x1 * np.log10(self.f)) + d1
        kV = 10**log_k1
        
        # Calculate aH (frequency-dependent parameter)
        log_k2 = 0
        a2 = [-0.14318, 0.29591, 0.32177, -5.37610, 16.1721]
        b2 = [1.82442, 0.77564, 0.63773, -0.96230, -3.29980]
        c2 = [-0.55187, 0.19822, 0.13164, 1.47828, 3.43990]
        x2 = 0.67849
        d2 = -1.95537
        
        for i in range(5):
            log_k2 += a2[i] * np.exp(-((np.log10(self.f) - b2[i]) / c2[i])**2)
        
        log_k2 += (x2 * np.log10(self.f)) + d2
        aH = log_k2
        
        # Calculate aV (frequency-dependent parameter)
        log_k3 = 0
        a3 = [-0.07771, 0.56727, -0.20238, -48.2991, 48.5833]
        b3 = [2.33840, 0.95545, 1.14520, 0.791669, 0.791459]
        c3 = [-0.76284, 0.54039, 0.26809, 0.116226, 0.116479]
        x3 = -0.053739
        d3 = 0.83433
        
        for n in range(5):
            log_k3 += a3[n] * np.exp(-((np.log10(self.f) - b3[n]) / c3[n])**2)
        
        log_k3 += (x3 * np.log10(self.f)) + d3
        aV = log_k3
        
        # Calculate polarization-dependent parameters
        y = np.cos(x)**2
        y1 = np.cos(2 * np.radians(45))
        y2 = y * y1
        
        k = (kH + kV + (kH - kV) * y2) / 2
        a = (kH * aH + kV * aV + (kH * aH - kV * aV) * y2) / (2 * k)
        
        # Calculate specific attenuation
        yr = k * self.r**a
        
        # Calculate horizontal reduction factor (hrf)
        m1 = lg * yr / self.f
        m2 = 0.38 * (1 - np.exp(-2 * lg))
        m3 = 0.78 * np.sqrt(m1 - m2) if (m1 - m2) > 0 else 0
        hrf = 1 / (1 + m3)
        
        # Calculate vertical adjustment factor
        z1 = (hr - self.hs) / (lg * hrf) if (lg * hrf) > 0 else 0
        z2 = np.arctan(z1)
        z = 1 / np.tan(z2) if z2 > 0 else float('inf')
        
        # Calculate effective path length
        if z > x:
            lr = lg * hrf / np.cos(x)
        else:
            lr = ls
        
        vrf = 1 / (1 + np.sqrt(np.sin(x)) * (31 * (1 - np.exp(-x)) * (np.sqrt(lr * yr) / self.f**2) - 0.45))
        
        # Calculate effective path length
        le = lr * vrf
        
        # Calculate predicted attenuation
        att = le * yr
        
        return abs(att)
    
    def calculate_gas_attenuation(self):
        """Calculate gas attenuation (oxygen and water vapor)"""
        # Water vapor density
        pw = 7.5  # water vapor density
        
        # Calculate parameters
        rt = 288 / self.T
        rp = self.p / 1013
        n1 = 0.955 * rp * (rt**0.68) + (0.006 * pw)
        n2 = 0.735 * rp * (rt**0.5) + (0.0353 * (rt**4) * pw)
        
        # Calculate the water vapor specific attenuation (Yw)
        f = self.f  # Frequency in GHz
        
        Yw = (((3.98 * n1 * np.exp(2.23 * (1 - rt))) / (((f - 22.235)**2) + 9.42 * (n1**2))) * (1 + ((f - 22) / (f + 22))**2) +
              ((11.96 * n1 * np.exp(0.7 * (1 - rt))) / (((f - 183.31)**2) + 11.14 * (n1**2))) +
              ((0.081 * n1 * np.exp(6.44 * (1 - rt))) / (((f - 321.226)**2) + (6.29 * (n1**2)))) +
              ((3.66 * n1 * np.exp(1.6 * (1 - rt))) / (((f - 325.153)**2) + 9.22 * (n1**2))) +
              ((25.37 * n1 * np.exp(1.09 * (1 - rt))) / ((f - 380)**2)) +
              ((17.4 * n1 * np.exp(1.46 * (1 - rt))) / ((f - 448)**2)) +
              ((844.6 * n1 * np.exp(0.17 * (1 - rt))) / ((f - 557)**2)) * (1 + ((f - 557) / (f + 557))**2) +
              ((290 * n1 * np.exp(0.41 * (1 - rt))) / ((f - 752)**2)) * (1 + ((f - 752) / (f + 752))**2) +
              ((83328 * n2 * np.exp(0.99 * (1 - rt))) / ((f - 1780)**2)) * (1 + ((f - 1780) / (f + 1780))**2)) * \
             ((f**2) * (rt**2.5) * (pw * 10**(1 - 5)))
        
        # Calculate the path length for water vapor contents (hw) (km)
        conw = 1.013 / (1 + np.exp((0 - 8.1) * (rp - 0.57)))
        hw = 1.66 * (1 + ((1.39 * conw) / (((f - 22.235)**2) + (2.56 * conw))) + 
                      ((3.37 * conw) / (((f - 183.31)**2) + (4.69 * conw))) + 
                      ((1.5 * conw) / (((f - 325.1)**2) + (2.89 * conw))))
        
        # Water vapor attenuation in zenith angle path (dB)
        Aw = Yw * hw
        
        # Calculate constants
        ee1 = (rp**0.0717) * (rt**(-1.8132)) * np.exp((0.0156 * (1 - rp)) + (-1.6515 * (1 - rt)))
        ee2 = (rp**0.5146) * (rt**(-4.6368)) * np.exp(((-0.1921) * (1 - rp)) + (-5.7416 * (1 - rt)))
        ee3 = (rp**0.3414) * (rt**(-6.5851)) * np.exp(((0.2130) * (1 - rp)) + (-8.5854 * (1 - rt)))
        
        # Calculate the dry air specific attenuation
        if f <= 54:
            Yo = (((7.2 * (rt**2.8)) / ((f**2) + (0.34 * (rp**2) * (rt**1.6)))) + 
                  ((0.62 * ee3) / (((54 - f)**(1.16 * ee1)) + (0.83 * ee2)))) * ((f**2) * (rp**2) * (10**(-3)))
        else:
            Yo = 0  # Add appropriate formula for f > 54 if needed
        
        # Calculate the equivalent height
        t1 = (4.64 / (1 + (0.066 * (rp**(-2.3))))) * np.exp(-(((f - 59.7) / (2.87 + (12.4 * np.exp((-7.9) * rp))))**2))
        t2 = (0.14 * np.exp(2.12 * rp)) / (((f - 118.75)**2) + 0.031 * np.exp(2.2 * rp))
        t3 = (0.0114 / (1 + (0.14 * (rp**(-2.6))))) * f * ((-0.0247 + (0.0001 * f) + (1.61 * (10**(-6)) * f**2)) / 
                                                          (1 - (0.0169 * f) + (4.1 * (10**(-5)) * f**2) + 
                                                           (3.2 * (10**(-7)) * f**3)))
        ho = (6.1 / (1 + (0.17 * (rp**(-1.1))))) * (1 + t1 + t2 + t3)
        
        # Dry air attenuation in zenith angle path (dB)
        Ao = Yo * ho
        
        # Calculate the total gases attenuation for the given elevation
        Atotx = (Ao + Aw) / np.sin(np.radians(self.Elev))
        
        return abs(Atotx)
    
    def calculate_tropospheric_scintillation(self):
        """Calculate tropospheric scintillation"""
        # Calculate the saturation water vapor pressure
        es = 6.1121 * np.exp((17.502 * self.T_celcius) / (self.T_celcius + 240.97))
        
        # Compute the wet term of the radio refractivity
        H = self.RH  # Relative humidity in percentage
        Nwet = 3732 * H * es / ((self.T_celcius + 273)**2)
        
        # Calculate the standard deviation of the signal amplitude
        ref = 3.6e-3 + (Nwet * 1e-4)  # in dB
        
        # Calculate the effective path length
        hL = 1000  # Height of turbulent layer in meters
        L = (2 * hL) / (np.sqrt(np.sin(np.radians(self.Elev))**2 + 2.35e-4 + np.sin(np.radians(self.Elev))))
        
        # Calculate the effective antenna diameter
        Deff = np.sqrt(self.antenna_efficiency) * self.D
        
        # Calculate the antenna averaging factor
        x = (1.22 * Deff**2 * self.f) / L
        g = np.sqrt(3.86 * (x**2 + 1)**(11/12) * np.sin((11/6) * np.arctan(1/x)) - 7.08 * x**(5/6))
        
        # Calculate the standard deviation of the signal for the period and propagation path
        g_x = (ref * self.f**(7/12) * g) / (np.sin(np.radians(self.Elev))**1.2)
        
        # Calculate the time percentage factor
        p = 0.01  # Time percentage in %
        a_p = -0.061 * (np.log10(p))**3 + 0.072 * (np.log10(p))**2 - 1.71 * np.log10(p) + 3.0
        
        # Calculate the scintillation fade depth
        TroScin = a_p * g_x
        
        return abs(TroScin)
    
    def calculate_ionospheric_scintillation(self):
        """Calculate ionospheric scintillation (only for f < 3 GHz)"""
        if self.f < 3:
            return 10.0  # Fixed value for frequencies < 3 GHz
        else:
            return 0.0  # No ionospheric scintillation for frequencies >= 3 GHz
    
    def calculate_total_attenuation(self):
        """Calculate the total atmospheric attenuation"""
        # Calculate individual components
        cloud_att = self.calculate_cloud_attenuation()
        rain_att = self.calculate_rain_attenuation()
        gas_att = self.calculate_gas_attenuation()
        trop_scin = self.calculate_tropospheric_scintillation()
        ion_scin = self.calculate_ionospheric_scintillation()
        
        # Sum all components
        total_att = cloud_att + rain_att + gas_att + trop_scin + ion_scin
        
        return total_att
    
    def get_detailed_results(self):
        """Get detailed results of all attenuation components"""
        cloud_att = self.calculate_cloud_attenuation()
        rain_att = self.calculate_rain_attenuation()
        gas_att = self.calculate_gas_attenuation()
        trop_scin = self.calculate_tropospheric_scintillation()
        ion_scin = self.calculate_ionospheric_scintillation()
        total_att = cloud_att + rain_att + gas_att + trop_scin + ion_scin
        
        results = {
            'frequency_GHz': self.f,
            'elevation_angle_deg': self.Elev,
            'cloud_attenuation_dB': cloud_att,
            'rain_attenuation_dB': rain_att,
            'gas_attenuation_dB': gas_att,
            'tropospheric_scintillation_dB': trop_scin,
            'ionospheric_scintillation_dB': ion_scin,
            'total_attenuation_dB': total_att
        }
        
        return results


# Example usage:
if __name__ == "__main__":
    # Create an instance with required parameters
    atm_loss = Atmospheric_loss(f_c=2.5, elevation_angle=30)
    
    # Calculate total attenuation
    total_att = atm_loss.calculate_total_attenuation()
    print(f"Total atmospheric attenuation: {total_att:.2f} dB")
    
    # Get detailed results
    results = atm_loss.get_detailed_results()
    for key, value in results.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
