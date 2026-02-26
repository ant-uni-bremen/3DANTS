import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0  # Import Bessel function
import math

class FadingSimulation:
    """
    Flat-fading channel simulator for LEO satellite links.

    Temporal-correlation / Doppler modes
    =====================================
    At S/Ka-band the true satellite Doppler is ~60-190 kHz, giving a channel
    coherence time Tc ~ 5-7 us (Clarke: Tc = 0.423 / f_D).  A system-level
    simulator running at ms-s resolution cannot observe this fast variation;
    each step already sees a fully decorrelated small-scale fading sample.

    What the simulator needs to capture is the *slow* temporal envelope
    variation arising from two physically motivated sources:

    Option A -- Residual Doppler after satellite pre-compensation  [physical]
        After a UE/GS applies standard NTN Doppler pre-compensation
        (3GPP TS 38.821), the dominant residual comes from UE movement:
            f_D_residual = v_UE * fc / c
        Pedestrian at 1.5 m/s -> f_D ~ 12.5 Hz -> Tc ~ 34 ms at 2.5 GHz.
        Use: doppler_mode='compensated',  v_residual_mps=<UE speed m/s>

    Option B -- Explicit coherence-time target  [honest scaling]
        Derive f_D from a desired Tc directly.  Useful for sensitivity studies:
            f_D = 0.423 / Tc_target_s   (Clarke's formula)
        Use: doppler_mode='scaled',  Tc_target_s=<seconds>,  fc_ref=<Hz>

    Option C -- Full satellite Doppler  [link-level / us-resolution use]
        True orbital velocity, Tc ~ 7 us.  Statistically correct when fs is
        in the MHz range; samples are IID at ms resolution.
        Use: doppler_mode='full'

    Backward compatibility
    ----------------------
    The legacy  Doppler_compensate='Yes'/'No'  keyword still works:
        'No'  -> doppler_mode='full'
        'Yes' -> doppler_mode='compensated', v_residual_mps=1.5 m/s
                  (replaces the old erroneous sqrt(GM/r^2) formula)

    Parameters
    ----------
    num_samples    : int   -- number of channel samples
    fs             : float -- sampling frequency [Hz]
    K              : float -- Rician K-factor (linear)
    N              : int   -- number of sinusoids in Jakes model
    h              : float -- satellite altitude [m]
    doppler_mode   : str   -- 'compensated' | 'scaled' | 'full'
    v_residual_mps : float -- UE residual speed [m/s]           (Option A)
    Tc_target_s    : float -- desired coherence time [s]        (Option B)
    fc_ref         : float -- carrier frequency for Tc->f_D [Hz](Option B)
    Doppler_compensate : str -- legacy keyword ('Yes'/'No')
    """

    def __init__(self, num_samples, fs, K, N, h,
                 doppler_mode='full',
                 v_residual_mps=1.5,
                 Tc_target_s=0.034,
                 fc_ref=2.5e9,
                 Doppler_compensate=None):

        self.num_samples = num_samples
        self.fs  = fs
        self.K   = K
        self.N   = N
        self.h   = h

        self.c = 3e8
        self.R = 6371e3
        self.G = 6.6743e-11
        self.M = 5.9722e24

        # True orbital velocity (vis-viva)
        self._v_orbital = math.sqrt(self.G * self.M / (self.R + self.h))

        # Legacy keyword mapping
        if Doppler_compensate is not None:
            doppler_mode = 'compensated' if Doppler_compensate == 'Yes' else 'full'

        if doppler_mode not in ('full', 'compensated', 'scaled'):
            raise ValueError(f"doppler_mode must be 'full', 'compensated', or 'scaled'. Got '{doppler_mode}'.")

        self.doppler_mode   = doppler_mode
        self.v_residual_mps = v_residual_mps
        self.Tc_target_s    = Tc_target_s
        self.fc_ref         = fc_ref

        if doppler_mode == 'full':
            # Option C: true orbital speed
            self.v_sat = self._v_orbital

        elif doppler_mode == 'compensated':
            # Option A: residual UE speed after satellite Doppler pre-compensation
            self.v_sat = v_residual_mps

        elif doppler_mode == 'scaled':
            # Option B: back-calculate v from desired coherence time
            # Clarke: Tc = 0.423 / f_D  =>  f_D = 0.423 / Tc  =>  v = f_D * c / fc_ref
            fd_target  = 0.423 / Tc_target_s
            self.v_sat = fd_target * self.c / fc_ref

        # Effective coherence time at fc_ref (for reference / logging)
        fd_ref = self.v_sat / self.c * fc_ref
        self.Tc_effective = 0.423 / fd_ref if fd_ref > 0 else float('inf')

    # ------------------------------------------------------------------
    def f_D(self, theta, fc):
        """
        Effective Doppler frequency [Hz].

        Full mode: uses the orbital geometry projection
            f_D = (v_orbital/c) * (R/(R+h)) * cos(theta) * fc
        Compensated / Scaled: residual Doppler is isotropic (UE direction
        is random), so theta is not used:
            f_D = (v_eff/c) * fc
        """
        if self.doppler_mode == 'full':
            return (self.v_sat / self.c) * (self.R / (self.R + self.h)) * np.cos(theta) * fc
        else:
            return (self.v_sat / self.c) * fc

    # ------------------------------------------------------------------
    def rician_fading_accurate(self, fd):
        """
        Temporally-correlated complex Rician fading (Jakes model).

        NLOS scatter normalised by N (not N+1) so that
        E[|Z_NLOS|^2] = 1/(1+K),  giving E[|Z_total|^2] = 1.
        """
        t       = np.arange(self.num_samples) / self.fs
        omega_m = 2 * np.pi * fd

        Z = np.zeros(self.num_samples, dtype=np.complex128)
        for n in range(1, self.N + 1):
            theta_n = 2 * np.pi * np.random.rand() - np.pi
            phi_n   = 2 * np.pi * np.random.rand() - np.pi
            Z += np.exp(1j * omega_m * t * np.cos(
                    (2 * np.pi * n + theta_n) / self.N)) * np.exp(1j * phi_n)

        Z *= np.sqrt(1 / (self.N * (1 + self.K)))   # N, not N+1

        theta_0 = np.pi / 4
        phi_0   = 2 * np.pi * np.random.rand() - np.pi
        Z_LOS   = np.sqrt(self.K / (1 + self.K)) * np.exp(
                    1j * (omega_m * t * np.cos(theta_0) + phi_0))

        return Z + Z_LOS

    # ------------------------------------------------------------------
    def b_0_calc(self, theta):
        return -4.7943e-8*theta**3 + 5.5784e-6*theta**2 - 2.1344e-4*theta + 3.2710e-2

    def m_theta(self, theta):
        return  6.3739e-5*theta**3 + 5.8533e-4*theta**2 - 1.5973e-1*theta + 3.5156

    def Omega_theta(self, theta):
        return  1.4428e-5*theta**3 - 2.3798e-3*theta**2 + 0.12702*theta  - 1.4864

    # ------------------------------------------------------------------
    def nakagami_m_based_Gamma(self, m, Omega):
        shape         = 2 * m
        scale         = Omega / m
        Gamma_samples = np.random.gamma(shape, scale, self.num_samples)
        return np.sqrt(Gamma_samples)

    def Rank_matching(self, rayleigh_seq, nakagami_seq):
        r_idx           = np.argsort(rayleigh_seq)
        n_idx           = np.argsort(nakagami_seq)
        nakagami_sorted = nakagami_seq[n_idx]
        matched         = np.empty_like(nakagami_sorted)
        matched[r_idx]  = nakagami_sorted
        return matched

    # ------------------------------------------------------------------
    def run_simulation(self, theta_degrees, fc_array, distance_satellite):
        if not isinstance(theta_degrees, (list, np.ndarray)):
            theta_degrees = [theta_degrees]
        if not isinstance(fc_array, (list, np.ndarray)):
            fc_array = [fc_array]

        theta_radians      = np.deg2rad(theta_degrees)
        distance_satellite = distance_satellite * 1000  # km -> m

        for fc in fc_array:
            for theta_deg, theta_rad in zip(theta_degrees, theta_radians):
                fd = self.f_D(theta_rad, fc)

                rayleigh_fading_samples = self.rician_fading_accurate(fd)

                b_0   = self.b_0_calc(theta_deg)
                m     = max(0.835,    self.m_theta(theta_deg))
                Omega = max(0.000897, self.Omega_theta(theta_deg))

                nakagami_sequence      = self.nakagami_m_based_Gamma(m, Omega)
                lower_doppler_rayleigh = self.rician_fading_accurate(fd / 100)
                nakagami_correlated    = self.Rank_matching(
                    np.abs(lower_doppler_rayleigh), nakagami_sequence)

                phase = np.exp(-1j * 2 * np.pi * distance_satellite * fc / self.c)
                Shadowed_rician_channel = (rayleigh_fading_samples
                                           + nakagami_correlated * phase)

        return Shadowed_rician_channel

    # ------------------------------------------------------------------
    def channel_with_UE_movement(self, theta_degrees, fc_array,
                                  UE_speed, UE_moving_direction,
                                  distance_satellite):
        """
        Channel where Doppler is driven by explicit UE speed and direction.
        This is always Option A semantics regardless of the instance's
        doppler_mode: the UE velocity overrides the instance v_sat.
        """
        if not isinstance(theta_degrees, (list, np.ndarray)):
            theta_degrees = [theta_degrees]
        if not isinstance(fc_array, (list, np.ndarray)):
            fc_array = [fc_array]

        theta_radians      = np.deg2rad(theta_degrees)
        distance_satellite = distance_satellite * 1000

        for fc in fc_array:
            for theta_deg, theta_rad in zip(theta_degrees, theta_radians):
                fd_sat = self.f_D(theta_rad, fc)
                # UE Doppler: includes satellite residual fd_sat in carrier
                fd_ue  = (UE_speed / self.c) * (fc + fd_sat) * np.cos(UE_moving_direction)

                rayleigh_fading_samples = self.rician_fading_accurate(fd_ue)

                b_0   = self.b_0_calc(theta_deg)
                m     = max(0.835,    self.m_theta(theta_deg))
                Omega = max(0.000897, self.Omega_theta(theta_deg))

                nakagami_sequence      = self.nakagami_m_based_Gamma(m, Omega)
                lower_doppler_rayleigh = self.rician_fading_accurate(fd_ue / 100)
                nakagami_correlated    = self.Rank_matching(
                    np.abs(lower_doppler_rayleigh), nakagami_sequence)

                phase = np.exp(-1j * 2 * np.pi * distance_satellite * fc / self.c)
                Shadowed_rician_channel = (rayleigh_fading_samples
                                           + nakagami_correlated * phase)

        return Shadowed_rician_channel
