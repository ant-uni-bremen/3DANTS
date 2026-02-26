#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file was created by the Department of Communications Engineering,
University of Bremen, Germany.
https://github.com/ant-uni-bremen
Copyright (c) 2026 Department of Communications Engineering, University of Bremen
SPDX-License-Identifier: Apache-2.0
"""
"""

Shadowed-Rician small-scale fading for LEO satellite links.

Channel model
-------------
The composite channel is the Loo / 3GPP TR 38.811 shadowed-Rician model:

    H[t] = H_rician[t]  +  H_nakagami[t] * exp(-j*2*pi*d*fc/c)

where
  H_rician   -- temporally-correlated Rician component (Jakes model)
                 driven by the effective Doppler fd (see below)
  H_nakagami -- Nakagami-m envelope, rank-matched to a low-Doppler
                 Rayleigh reference to impose slow temporal correlation
  (b0, m, Omega, K) always come from the 3GPP TR 38.811 elevation-angle
                 polynomial fits, so the *marginal distribution* is always
                 geometry-driven regardless of the Doppler mode.

Doppler modes
-------------
Three modes control how fd is computed.  The choice affects ONLY the
temporal correlation structure; it does not change the marginal distribution.

  'full'        -- fd = |v_LOS| * fc / c
                   True radial velocity from the real orbit trajectory.
                   Coherence time Tc ~ 7 us at S-band.
                   Statistically correct when simulation runs at µs resolution.
                   At ms resolution, samples within a batch are essentially IID,
                   which is also statistically correct (channel decorrelates
                   ~140 times per 1-ms step).

  'compensated' -- fd = v_residual_mps * fc / c
                   Residual Doppler seen by a UE after satellite Doppler
                   pre-compensation (3GPP TS 38.821).  Dominated by UE movement.
                   Pedestrian 1.5 m/s -> fd ~ 12.5 Hz -> Tc ~ 34 ms at 2.5 GHz.
                   This gives meaningful temporal correlation at ms resolution.

  'scaled'      -- fd = 0.423 / Tc_target_s  (Clarke's formula)
                   Equivalent speed back-calculated from a desired coherence time.
                   Useful for sensitivity studies.

In all three modes, v_LOS is still passed to run_simulation so that the
calling code (e.g. 3D_network_with_traffic.py) can log it and use it for
other geometry computations.  Only the internal fd used by the Jakes model
changes between modes.

Bugs fixed vs original
-----------------------
  B1  Rank_matching now receives np.abs(lower_doppler_rayleigh) -- the
      original passed the raw complex array, causing argsort to sort by
      real part only (wrong rank order).
  B2  self.fs is no longer mutated inside run_simulation.  A local fs_jakes
      is computed for Jakes sampling without side-effects on the instance.
"""

import numpy as np
import math


class Satellite_Fading_channel:
    """
    Parameters
    ----------
    num_samples_nakagami : int
        Total samples per batch (Nakagami envelope length).
    num_samples_rician : int
        Samples per Rician segment.  num_segments = num_samples_nakagami //
        num_samples_rician.  Typical: num_samples_rician = batch // 10.
    fs : float
        Nominal sampling frequency [Hz].  Used as the outer-loop time step
        (e.g. 1000 Hz = 1 ms/sample).  Not mutated by run_simulation.
    N : int
        Number of sinusoids in the Jakes model.
    doppler_mode : str
        'full' | 'compensated' | 'scaled'  (default 'full').
    v_residual_mps : float
        UE residual speed [m/s] for mode='compensated' (default 1.5 m/s).
    Tc_target_s : float
        Desired coherence time [s] for mode='scaled' (default 0.034 s = 34 ms).
    fc_ref : float
        Carrier frequency [Hz] used to convert Tc_target_s -> fd for
        mode='scaled' (default 2.5e9 Hz).
    """

    def __init__(self, num_samples_nakagami, num_samples_rician, fs, N,
                 doppler_mode='full',
                 v_residual_mps=1.5,
                 Tc_target_s=0.034,
                 fc_ref=2.5e9):

        self.num_samples_nakagami = num_samples_nakagami
        self.num_samples_rician   = num_samples_rician
        self.num_segments         = int(num_samples_nakagami / num_samples_rician)
        self.fs                   = fs   # never mutated
        self.N                    = N

        # Physical constants
        self.c = 3.0e8

        # Validate and store Doppler mode
        if doppler_mode not in ('full', 'compensated', 'scaled'):
            raise ValueError(
                f"doppler_mode must be 'full', 'compensated', or 'scaled'. "
                f"Got '{doppler_mode}'.")
        self.doppler_mode    = doppler_mode
        self.v_residual_mps  = v_residual_mps
        self.Tc_target_s     = Tc_target_s
        self.fc_ref          = fc_ref

        # Pre-compute the effective speed for compensated/scaled modes
        # (fd will be computed per-call using the actual fc)
        if doppler_mode == 'compensated':
            self._v_eff = v_residual_mps              # m/s
        elif doppler_mode == 'scaled':
            fd_target   = 0.423 / Tc_target_s        # Clarke: Tc = 0.423/fd
            self._v_eff = fd_target * self.c / fc_ref  # equivalent speed [m/s]
        else:
            self._v_eff = None  # not used; fd computed from v_LOS per call

    # ------------------------------------------------------------------
    def _effective_fd(self, v_LOS_kmps, fc):
        """
        Compute the effective Doppler frequency [Hz] for the Jakes model.

        Parameters
        ----------
        v_LOS_kmps : float
            Radial velocity from orbit trajectory [km/s].
            = dot(v_sat_vector, r_hat) where r_hat = (sat - gs) / |sat - gs|
        fc : float
            Carrier frequency [Hz].
        """
        if self.doppler_mode == 'full':
            # True satellite Doppler from range-rate
            fd = (np.abs(v_LOS_kmps) * 1000.0 / self.c) * fc
        else:
            # Compensated or scaled: use pre-computed effective speed
            fd = (self._v_eff / self.c) * fc
        return fd

    # ------------------------------------------------------------------
    def rician_fading_accurate(self, num_samples, fd, K, theta_rad):
        """
        Temporally-correlated complex Rician fading (Jakes / Clarke model).

        Normalisation: empirical (Z /= sqrt(mean(|Z|^2))) so that the NLOS
        power is exactly 1/(1+K) regardless of N.  More accurate than the
        analytic 1/sqrt(N*(1+K)) factor for small N.

        The Jakes model is sampled at fs_jakes = max(fs, ceil(2*fd)) to
        satisfy Nyquist internally.  This local fs_jakes does NOT mutate
        self.fs (bug B2 fixed).

        Parameters
        ----------
        num_samples : int   -- length of output
        fd          : float -- Doppler frequency [Hz]
        K           : float -- Rician K-factor (linear)
        theta_rad   : float -- LOS arrival angle [rad] (elevation)
        """
        # Local Nyquist-compliant sampling rate -- does NOT touch self.fs
        fs_jakes = max(self.fs, math.ceil(2.0 * fd)) if fd > 0 else self.fs
        t        = np.arange(num_samples) / fs_jakes
        omega_m  = 2.0 * np.pi * fd

        Z = np.zeros(num_samples, dtype=np.complex128)
        for n in range(1, self.N + 1):
            theta_n = 2.0 * np.pi * np.random.rand() - np.pi
            phi_n   = 2.0 * np.pi * np.random.rand() - np.pi
            Z += np.exp(1j * omega_m * t * np.cos(
                    (2.0 * np.pi * n + theta_n) / self.N)) * np.exp(1j * phi_n)

        # Empirical normalisation: correct regardless of N
        power = np.mean(np.abs(Z) ** 2)
        if power > 0:
            Z /= np.sqrt(power)
        Z *= np.sqrt(1.0 / (1.0 + K))

        # LOS component at elevation angle
        phi_0 = 2.0 * np.pi * np.random.rand() - np.pi
        Z_LOS = np.sqrt(K / (1.0 + K)) * np.exp(
                    1j * (omega_m * t * np.cos(theta_rad) + phi_0))

        return Z + Z_LOS

    # ------------------------------------------------------------------
    def b_0_calc(self, theta):
        theta = max(theta, 17.8)  # polynomial valid for theta >= ~18 deg
        return -4.7943e-8*theta**3 + 5.5784e-6*theta**2 - 2.1344e-4*theta + 3.2710e-2

    def m_theta(self, theta):
        theta = max(theta, 17.8)
        return  6.3739e-5*theta**3 + 5.8533e-4*theta**2 - 1.5973e-1*theta + 3.5156

    def Omega_theta(self, theta):
        theta = max(theta, 17.8)
        return  1.4428e-5*theta**3 - 2.3798e-3*theta**2 + 0.12702*theta  - 1.4864

    # ------------------------------------------------------------------
    def nakagami_m_based_Gamma(self, m, Omega):
        shape         = 2.0 * m
        scale         = Omega / m
        Gamma_samples = np.random.gamma(shape, scale, self.num_samples_nakagami)
        return np.sqrt(Gamma_samples)

    # ------------------------------------------------------------------
    def Rank_matching(self, rayleigh_envelope, nakagami_seq):
        """
        Impose the rank-order structure of rayleigh_envelope onto nakagami_seq.

        Parameters
        ----------
        rayleigh_envelope : real 1-D array  -- |h_rayleigh| (envelope, NOT complex)
        nakagami_seq      : real 1-D array  -- Nakagami-m amplitude samples

        Note: B1 fix -- caller must pass np.abs(rayleigh), not raw complex.
        This method asserts real input to catch the bug at the boundary.
        """
        if np.iscomplexobj(rayleigh_envelope):
            raise TypeError(
                "Rank_matching: rayleigh_envelope must be real (envelope). "
                "Pass np.abs(rayleigh_fading_samples) instead of the raw complex array.")
        r_idx           = np.argsort(rayleigh_envelope)
        n_idx           = np.argsort(nakagami_seq)
        nakagami_sorted = nakagami_seq[n_idx]
        matched         = np.empty_like(nakagami_sorted)
        matched[r_idx]  = nakagami_sorted
        return matched

    # ------------------------------------------------------------------
    def run_simulation(self, theta_degrees, v_LEO, fc_array, distance_satellite):
        """
        Generate a batch of shadowed-Rician channel samples.

        Parameters
        ----------
        theta_degrees      : float or list -- elevation angle(s) [degrees]
        v_LEO              : float         -- radial (LOS) velocity [km/s]
                             = dot(v_sat_vector, r_hat) from the real orbit.
                             Used as fd source in 'full' mode.
                             Always passed so callers can log it; ignored for
                             fd in 'compensated' and 'scaled' modes.
        fc_array           : float or list -- carrier frequency/ies [Hz]
        distance_satellite : float         -- slant range [km]

        Returns
        -------
        Shadowed_rician_channel : complex ndarray of length num_samples_nakagami
        """
        if not isinstance(theta_degrees, (list, np.ndarray)):
            theta_degrees = [theta_degrees]
        if not isinstance(fc_array, (list, np.ndarray)):
            fc_array = [fc_array]

        theta_radians      = np.deg2rad(theta_degrees)
        distance_m         = distance_satellite * 1000.0  # km -> m

        for fc in fc_array:
            for theta_deg, theta_rad in zip(theta_degrees, theta_radians):

                # ── 3GPP TR 38.811 polynomials valid for theta >= ~18° ──────
                if theta_deg < 20.0:
                    theta_deg = 20.0
                    theta_rad = np.deg2rad(20.0)

                # ── Effective Doppler frequency ──────────────────────────────
                # In 'full' mode: true satellite Doppler from range-rate.
                # In 'compensated': UE residual after pre-compensation.
                # In 'scaled': derived from target coherence time.
                # The marginal distribution (b0, m, Omega, K) is always
                # geometry-driven regardless of mode.
                fd = self._effective_fd(v_LEO, fc)

                # ── Channel distribution parameters from elevation angle ──────
                b_0   = self.b_0_calc(theta_deg)
                m     = self.m_theta(theta_deg)
                Omega = self.Omega_theta(theta_deg)
                K     = Omega / (2.0 * b_0)   # Loo model K-factor

                # ── Rician component: num_segments batches concatenated ───────
                # Each segment is independently generated (new random phases)
                # so that the batch covers the full num_samples_nakagami length
                # while keeping each Jakes realisation short (lower memory,
                # avoids aliasing artefacts accumulating over long sequences).
                rician_segments = []
                for _ in range(self.num_segments):
                    seg = self.rician_fading_accurate(
                        self.num_samples_rician, fd, K, theta_rad)
                    rician_segments.append(seg)
                rician_fading_samples = np.concatenate(rician_segments)

                # ── Nakagami-m component with rank-matched correlation ────────
                nakagami_sequence = self.nakagami_m_based_Gamma(m, Omega)

                # Generate a slow-varying Rayleigh reference (fd/100) to impose
                # temporal structure onto the Nakagami envelope via rank matching.
                # fd/100: reduces Doppler by 100x -> Tc 100x longer.
                # At ms resolution this gives visible sample-to-sample correlation.
                lower_doppler_rayleigh = self.rician_fading_accurate(
                    self.num_samples_nakagami, fd / 100.0, K=0, theta_rad=theta_rad)

                # B1 FIX: pass envelope (abs), not raw complex
                nakagami_correlated = self.Rank_matching(
                    np.abs(lower_doppler_rayleigh),   # real envelope
                    nakagami_sequence)

                # ── Combine: shadowed Rician channel ─────────────────────────
                phase = np.exp(-1j * 2.0 * np.pi * distance_m * fc / self.c)
                Shadowed_rician_channel = rician_fading_samples + nakagami_correlated * phase

        return Shadowed_rician_channel
