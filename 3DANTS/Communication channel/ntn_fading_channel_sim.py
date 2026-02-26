#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file was created by the Department of Communications Engineering,
University of Bremen, Germany.
https://github.com/ant-uni-bremen
Copyright (c) 2026 Department of Communications Engineering, University of Bremen
SPDX-License-Identifier: Apache-2.0
"""

import numpy as np

class FadingSimulation_Non_terrestrial:
    """
    Shadowed-Rician channel simulator for NTN satellite links.

    Temporal-correlation / Doppler modes
    =====================================
    Identical three-mode design as FadingSimulation.  See that class for
    the full rationale.  Summary:

    Option A -- doppler_mode='compensated', v_residual_mps=<UE speed m/s>
        Residual Doppler after satellite pre-compensation driven by UE speed.
        Pedestrian 1.5 m/s -> Tc ~ 34 ms at 2.5 GHz.

    Option B -- doppler_mode='scaled', Tc_target_s=<s>, fc_ref=<Hz>
        Back-calculates v from Clarke: f_D = 0.423/Tc, v = f_D*c/fc_ref.

    Option C -- doppler_mode='full'
        True orbital velocity, Tc ~ 7 us. Use at us-resolution.

    Legacy: Doppler_compensate='Yes'/'No' still accepted.
    """

    def __init__(self, num_samples, fs, N, h,
                 doppler_mode='full',
                 v_residual_mps=1.5,
                 Tc_target_s=0.034,
                 fc_ref=2.5e9,
                 Doppler_compensate="compensated"):

        self.num_samples = num_samples
        self.fs = fs
        self.N  = N
        self.h  = h

        self.c = 3e8
        self.R = 6371e3
        self.G = 6.6743e-11
        self.M = 5.9722e24

        self._v_orbital = (self.G * self.M / (self.R + self.h)) ** 0.5

        # Legacy mapping
        if Doppler_compensate is not None:
            doppler_mode = 'compensated' if Doppler_compensate == 'Yes' else 'full'

        if doppler_mode not in ('full', 'compensated', 'scaled'):
            raise ValueError(f"doppler_mode must be 'full', 'compensated', or 'scaled'. Got '{doppler_mode}'.")

        self.doppler_mode   = doppler_mode
        self.v_residual_mps = v_residual_mps
        self.Tc_target_s    = Tc_target_s
        self.fc_ref         = fc_ref

        if doppler_mode == 'full':
            self.v_sat = self._v_orbital
        elif doppler_mode == 'compensated':
            self.v_sat = v_residual_mps
        elif doppler_mode == 'scaled':
            fd_target  = 0.423 / Tc_target_s
            self.v_sat = fd_target * self.c / fc_ref

        fd_ref = self.v_sat / self.c * fc_ref
        self.Tc_effective = 0.423 / fd_ref if fd_ref > 0 else float('inf')

    # ------------------------------------------------------------------
    def b_0_calc(self, theta):
        return -4.7943e-8*theta**3 + 5.5784e-6*theta**2 - 2.1344e-4*theta + 3.2710e-2

    def m_theta(self, theta):
        return  6.3739e-5*theta**3 + 5.8533e-4*theta**2 - 1.5973e-1*theta + 3.5156

    def Omega_theta(self, theta):
        return  1.4428e-5*theta**3 - 2.3798e-3*theta**2 + 0.12702*theta  - 1.4864

    # ------------------------------------------------------------------
    def f_D(self, theta, fc):
        """
        Effective Doppler [Hz].  In full mode uses orbital geometry projection;
        in compensated/scaled mode is scalar (UE direction is random).
        """
        if self.doppler_mode == 'full':
            return (self.v_sat / self.c) * (self.R / (self.R + self.h)) * np.cos(theta) * fc
        else:
            return (self.v_sat / self.c) * fc

    # ------------------------------------------------------------------
    def rician_fading_accurate(self, b_0, fd):
        """
        Temporally-correlated Rayleigh scatter component (no LOS term here;
        the LOS/Nakagami component is combined in run_simulation).

        Normalised by N (not N+1): E[|Z|^2] = 2*b_0.
        """
        t       = np.arange(self.num_samples) / self.fs
        omega_m = 2 * np.pi * fd

        Z = np.zeros(self.num_samples, dtype=np.complex128)
        for n in range(1, self.N + 1):
            theta_n = 2 * np.pi * np.random.rand() - np.pi
            phi_n   = 2 * np.pi * np.random.rand() - np.pi
            Z += np.exp(1j * omega_m * t * np.cos(
                    (2 * np.pi * n + theta_n) / self.N)) * np.exp(1j * phi_n)

        Z *= np.sqrt(1 / self.N) * np.sqrt(2 * b_0)   # N, not N+1
        return Z

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
                fd  = self.f_D(theta_rad, fc)
                b_0 = self.b_0_calc(theta_deg)

                rayleigh_fading_samples = self.rician_fading_accurate(b_0, fd)

                m     = max(0.835,    self.m_theta(theta_deg))
                Omega = max(0.000897, self.Omega_theta(theta_deg))

                nakagami_sequence      = self.nakagami_m_based_Gamma(m, Omega)
                lower_doppler_rayleigh = self.rician_fading_accurate(b_0, fd / 100)
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
        if not isinstance(theta_degrees, (list, np.ndarray)):
            theta_degrees = [theta_degrees]
        if not isinstance(fc_array, (list, np.ndarray)):
            fc_array = [fc_array]

        theta_radians      = np.deg2rad(theta_degrees)
        distance_satellite = distance_satellite * 1000

        for fc in fc_array:
            for theta_deg, theta_rad in zip(theta_degrees, theta_radians):
                fd_sat = self.f_D(theta_rad, fc)
                fd_ue  = (UE_speed / self.c) * (fc + fd_sat) * np.cos(UE_moving_direction)

                b_0 = self.b_0_calc(theta_deg)
                rayleigh_fading_samples = self.rician_fading_accurate(b_0, fd_ue)

                m     = max(0.835,    self.m_theta(theta_deg))
                Omega = max(0.000897, self.Omega_theta(theta_deg))

                nakagami_sequence      = self.nakagami_m_based_Gamma(m, Omega)
                lower_doppler_rayleigh = self.rician_fading_accurate(b_0, fd_ue / 100)
                nakagami_correlated    = self.Rank_matching(
                    np.abs(lower_doppler_rayleigh), nakagami_sequence)

                phase = np.exp(-1j * 2 * np.pi * distance_satellite * fc / self.c)
                Shadowed_rician_channel = (rayleigh_fading_samples
                                           + nakagami_correlated * phase)

        return Shadowed_rician_channel
