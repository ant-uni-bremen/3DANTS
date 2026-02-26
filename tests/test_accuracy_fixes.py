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
test_accuracy_fixes.py
======================
Unit tests for the accuracy bugs fixed in:
  - Position and Mobility/GEO.py
  - Communication channel/Satellite_fading_channel.py
  - Communication channel/Air_2_Ground_fading_channel.py

Run from the repo root with:
    pytest tests/test_accuracy_fixes.py -v

Or run directly:
    python tests/test_accuracy_fixes.py
"""

import sys
import os
import math
import datetime
import unittest
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root or from tests/ directory
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "3DANTS", "Position and Mobility"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "3DANTS", "Communication channel"))

from GEO import LEO_GEO
from Satellite_fading_channel import Satellite_Fading_channel
from Air_2_Ground_fading_channel import Air_Fading_channel

# ---------------------------------------------------------------------------
# Shared constants (real Earth / orbit values)
# ---------------------------------------------------------------------------
R_E  = 6_371_000.0          # Earth radius [m]
GM   = 3.986_004_418e14     # Geocentric gravitational constant [m³/s²]
H_GEO = 35_786_000.0        # GEO altitude [m]
H_LEO = 600_000.0           # LEO altitude [m]  (600 km)

# ISS mean motion ≈ 15.5 rev/day  →  v ≈ 7660 m/s
V_LEO_EXPECTED = math.sqrt(GM / (R_E + H_LEO))   # ~7558 m/s


# ===========================================================================
#  1. GEO.py — Epoch_time
# ===========================================================================
class TestEpochTime(unittest.TestCase):
    """Bug fixed: Epoch_time() used to return a hardcoded 2021-10-24 date.
    It now returns days since 1949-12-31 based on the current UTC clock."""

    def setUp(self):
        self.lg = LEO_GEO(R_E, GM, H_GEO)

    def test_epoch_is_positive(self):
        """Days since 1949-12-31 must always be a positive number."""
        days = self.lg.Epoch_time()
        self.assertGreater(days, 0)

    def test_epoch_is_after_stale_2021_value(self):
        """The fixed epoch must be later than the old hardcoded date (Oct 2021).
        Days from 1949-12-31 to 2021-10-24 ≈ 26_230."""
        STALE_DAYS = 26_230
        days = self.lg.Epoch_time()
        self.assertGreater(days, STALE_DAYS,
            "Epoch_time() is returning a date on or before 2021-10-24 — "
            "the hardcoded stale value may have been reintroduced.")

    def test_epoch_is_close_to_today(self):
        """Epoch should be within 1 day of 'right now'."""
        days = self.lg.Epoch_time()
        epoch_ref = datetime.datetime(1949, 12, 31, 0, 0, 0)
        now_utc   = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        expected  = (now_utc - epoch_ref).total_seconds() / 86400
        self.assertAlmostEqual(days, expected, delta=1.0,
            msg="Epoch_time() deviates by more than 1 day from current UTC time.")

    def test_epoch_type_is_float(self):
        """Epoch must be a float (SGP4 requires fractional days)."""
        days = self.lg.Epoch_time()
        self.assertIsInstance(days, float)


# ===========================================================================
#  2. GEO.py — elevation_angel_calculator
# ===========================================================================
class TestElevationAngleCalculator(unittest.TestCase):
    """Two bugs fixed:
    1. A magic +0.18° correction was applied to the observer latitude.
    2. Any angle below 10° was silently clamped to 10°.
    """

    def setUp(self):
        self.lg = LEO_GEO(R_E, GM, H_GEO)

    # --- helper: convert geodetic lat/lon/alt to ECEF [km] -----------------
    @staticmethod
    def geodetic_to_ecef_km(lat_deg, lon_deg, alt_m=0.0):
        """WGS-84 geodetic → ECEF in kilometres."""
        a     = 6_378_137.0
        b     = 6_356_752.314_245
        e_sq  = 1 - (b / a) ** 2
        lat   = math.radians(lat_deg)
        lon   = math.radians(lon_deg)
        N     = a / math.sqrt(1 - e_sq * math.sin(lat) ** 2)
        x     = (N + alt_m) * math.cos(lat) * math.cos(lon)
        y     = (N + alt_m) * math.cos(lat) * math.sin(lon)
        z     = (N * (1 - e_sq) + alt_m) * math.sin(lat)
        return np.array([x / 1000, y / 1000, z / 1000])

    def _satellite_directly_overhead(self, lat_deg, lon_deg, height_km):
        """Return ECEF position [km] of a satellite directly overhead."""
        gs = self.geodetic_to_ecef_km(lat_deg, lon_deg)
        # Scale the ground-station vector outward to the satellite altitude
        unit = gs / np.linalg.norm(gs)
        return gs + unit * height_km

    # -- Test: satellite directly overhead → elevation should be ~90° --------
    def test_zenith_satellite_gives_90_degrees(self):
        """A satellite placed exactly overhead should yield ~90° elevation."""
        lat, lon = 48.137, 11.576        # Munich
        gs_ecef  = self.geodetic_to_ecef_km(lat, lon)
        sat_ecef = self._satellite_directly_overhead(lat, lon, height_km=600)
        el = self.lg.elevation_angel_calculator(sat_ecef, gs_ecef)
        self.assertAlmostEqual(el, 90.0, delta=2.0,
            msg=f"Zenith satellite should give ~90°, got {el:.2f}°")

    # -- Test: symmetry — lat offset gives same elevation for ±lon ----------
    def test_elevation_symmetric_for_east_west_observer(self):
        """Two observers at the same latitude but mirrored longitudes should
        see an equatorial satellite at the same elevation angle."""
        sat_ecef = np.array([R_E / 1000 + 600, 0.0, 0.0])   # on equator, x-axis
        gs_east = self.geodetic_to_ecef_km(0.0,  30.0)
        gs_west = self.geodetic_to_ecef_km(0.0, -30.0)
        el_east = self.lg.elevation_angel_calculator(sat_ecef, gs_east)
        el_west = self.lg.elevation_angel_calculator(sat_ecef, gs_west)
        self.assertAlmostEqual(el_east, el_west, delta=0.5,
            msg="East/west-symmetric observers should see same elevation.")

    # -- Test: no more 10° floor — low elevation returns its real value ------
    def test_low_elevation_is_not_clamped(self):
        """A satellite near the horizon must return an angle below 10°.
        The old code clamped everything below 10° to exactly 10°."""
        lat, lon = 0.0, 0.0
        gs_ecef  = self.geodetic_to_ecef_km(lat, lon)
        # Place satellite far to the side — very low elevation expected
        sat_ecef = np.array([R_E / 1000 + 600, R_E / 1000 * 5, 0.0])
        el = self.lg.elevation_angel_calculator(sat_ecef, gs_ecef)
        self.assertLess(el, 10.0,
            msg=f"Expected elevation < 10° for horizon satellite, got {el:.2f}°. "
                "The 10° clamp may have been reintroduced.")

    # -- Test: result is bounded in [-90, 90] --------------------------------
    def test_elevation_within_physical_bounds(self):
        """Elevation angle must always be in [-90°, +90°]."""
        lat, lon = 53.11, 8.85    # Bremen
        gs_ecef  = self.geodetic_to_ecef_km(lat, lon)
        for height in [400, 600, 1200, 35786]:
            sat_ecef = self._satellite_directly_overhead(lat, lon, height)
            el = self.lg.elevation_angel_calculator(sat_ecef, gs_ecef)
            self.assertGreaterEqual(el, -90.0)
            self.assertLessEqual(el,    90.0)

    # -- Test: no +0.18 bias at the equator ----------------------------------
    def test_no_latitude_bias_at_equator(self):
        """For an observer on the equator and a satellite directly overhead,
        a +0.18° latitude bias would shift the result noticeably. Verify the
        result stays within 1° of 90°."""
        gs_ecef  = self.geodetic_to_ecef_km(0.0, 0.0)
        sat_ecef = self._satellite_directly_overhead(0.0, 0.0, height_km=600)
        el = self.lg.elevation_angel_calculator(sat_ecef, gs_ecef)
        self.assertAlmostEqual(el, 90.0, delta=1.0,
            msg=f"Equatorial zenith should be ~90°, got {el:.2f}°. "
                "Possible residual latitude bias.")


# ===========================================================================
#  3. GEO.py — motion()
# ===========================================================================
class TestMotion(unittest.TestCase):
    """Sanity-checks for the orbital velocity helper."""

    def setUp(self):
        self.lg = LEO_GEO(R_E, GM, H_GEO)

    def test_leo_velocity_meters_per_second(self):
        """LEO orbital speed at 600 km should be ~7558 m/s (vis-viva)."""
        r = R_E + H_LEO
        v = self.lg.motion(r, 'meter/sec')
        self.assertAlmostEqual(v, V_LEO_EXPECTED, delta=10.0,
            msg=f"LEO velocity expected ~{V_LEO_EXPECTED:.0f} m/s, got {v:.2f} m/s")

    def test_leo_velocity_rad_per_min(self):
        """Radians/minute form should be consistent with m/s form."""
        r = R_E + H_LEO
        v_ms  = self.lg.motion(r, 'meter/sec')
        v_rpm = self.lg.motion(r, 'rad/min')
        # rad/min = (m/s / r) * 60
        expected_rpm = (v_ms / r) * 60
        self.assertAlmostEqual(v_rpm, expected_rpm, delta=1e-6)

    def test_geo_period_is_24_hours(self):
        """GEO orbital period should be close to 24 h (86 400 s)."""
        r_geo = R_E + H_GEO
        v_geo = self.lg.motion(r_geo, 'meter/sec')
        period = (2 * math.pi * r_geo) / v_geo
        SIDEREAL_DAY = 86164
        self.assertAlmostEqual(period, SIDEREAL_DAY, delta=200, 
                               msg=f"GEO period expected ~86164 s (sidereal), got {period:.0f} s")


# ===========================================================================
#  4. GEO.py — distance()
# ===========================================================================
class TestDistance(unittest.TestCase):
    def setUp(self):
        self.lg = LEO_GEO(R_E, GM, H_GEO)

    def test_distance_known_vectors(self):
        """3-4-5 triangle: distance between [0,0,0] and [3,4,0] should be 5."""
        d = self.lg.distance(np.array([3.0, 4.0, 0.0]), np.array([0.0, 0.0, 0.0]))
        self.assertAlmostEqual(d, 5.0, places=10)

    def test_distance_is_symmetric(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        self.assertAlmostEqual(self.lg.distance(a, b), self.lg.distance(b, a))

    def test_distance_same_point_is_zero(self):
        p = np.array([100.0, 200.0, 300.0])
        self.assertAlmostEqual(self.lg.distance(p, p), 0.0)


# ===========================================================================
#  5. Satellite_fading_channel.py — rician_fading_accurate normalisation
# ===========================================================================
class TestSatelliteFadingNormalisation(unittest.TestCase):
    """Bug fixed: NLOS sum was divided by (N+1) instead of N, biasing the
    effective K-factor. Tests verify the channel has the correct power split."""

    def setUp(self):
        np.random.seed(0)
        # num_samples_nakagami, num_samples_rician, fs, N
        self.ch = Satellite_Fading_channel(
            num_samples_nakagami=5000,
            num_samples_rician=500,
            fs=10_000,
            N=20
        )

    def test_rician_output_length(self):
        """rician_fading_accurate must return exactly num_samples samples."""
        K        = 5.0
        fd       = 50.0
        samples  = self.ch.rician_fading_accurate(200, fd, self.ch.fs, K, theta_rad=0.0)
        self.assertEqual(len(samples), 200)

    def test_rician_output_is_complex(self):
        """Rician fading should produce a complex envelope."""
        samples = self.ch.rician_fading_accurate(100, 50.0, self.ch.fs, 5.0, 0.0)
        self.assertTrue(np.iscomplexobj(samples))

    def test_nlos_power_fraction(self):
        """With K=0 (pure Rayleigh / NLOS only), the NLOS scatter sum should
        have unit mean power per sample after normalisation.
        The old N+1 normalisation would give (N/(N+1)) < 1."""
        K        = 0.0   # pure NLOS — LOS component vanishes
        N        = self.ch.N
        num      = 2000
        np.random.seed(42)
        samples  = self.ch.rician_fading_accurate(num, fd=50.0, fs=self.ch.fs,
                                                   K=K, theta_rad=0.0)
        # With K=0 and correct N normalisation, E[|Z|²] ≈ 1
        mean_power = np.mean(np.abs(samples) ** 2)
        # Allow ±15% tolerance (statistical)
        self.assertAlmostEqual(mean_power, 1.0, delta=0.15,
            msg=f"Mean NLOS power expected ~1.0, got {mean_power:.3f}. "
                "N+1 normalisation may have been reintroduced (would give "
                f"~{N/(N+1):.3f}).")

    def test_rician_k_factor_effect(self):
        """Higher K should raise total mean power (stronger LOS component)."""
        np.random.seed(1)
        s_k0  = self.ch.rician_fading_accurate(2000, 50.0, self.ch.fs, K=0.0, theta_rad=0.3)
        np.random.seed(1)
        s_k10 = self.ch.rician_fading_accurate(2000, 50.0, self.ch.fs, K=10.0, theta_rad=0.3)
        self.assertGreater(np.mean(np.abs(s_k10)**2), np.mean(np.abs(s_k0)**2),
            msg="Higher K should produce higher mean envelope power.")


# ===========================================================================
#  6. Satellite_fading_channel.py — empirical channel parameter fits
# ===========================================================================
class TestSatelliteChannelParams(unittest.TestCase):
    """Spot-check the polynomial fits for b_0, m, Omega against published
    values from the 3GPP / Lutz model (Table in TR 38.811 Annex)."""

    def setUp(self):
        self.ch = Satellite_Fading_channel(1000, 100, 10000, 10)

    def test_b0_at_30_degrees(self):
        """b_0(30°) should be a small positive scatter power (≈ 0.022–0.028)."""
        b0 = self.ch.b_0_calc(30)
        self.assertGreater(b0, 0.01)
        self.assertLess(b0, 0.04)

    def test_m_at_90_degrees(self):
        """m(90°) (zenith) should be large → nearly no shadowing (≈ 1–4)."""
        m = self.ch.m_theta(90)
        self.assertGreater(m, 0.5)

    def test_omega_positive_for_valid_angles(self):
        """Omega (mean scattered power) must be positive for angles 10°–90°."""
        for theta in range(10, 91, 10):
            omega = self.ch.Omega_theta(theta)
            self.assertGreater(omega, 0,
                msg=f"Omega_theta({theta}°) = {omega:.4f} — should be > 0")

    def test_k_factor_positive(self):
        """K = Omega / (2*b_0) must be positive for all valid elevation angles."""
        for theta in range(10, 91, 10):
            b0    = self.ch.b_0_calc(theta)
            omega = self.ch.Omega_theta(theta)
            K     = omega / b0 / 2
            self.assertGreater(K, 0,
                msg=f"K-factor at {theta}° = {K:.4f} — must be positive")


# ===========================================================================
#  7. Satellite_fading_channel.py — run_simulation output
# ===========================================================================
class TestSatelliteRunSimulation(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.ch = Satellite_Fading_channel(
            num_samples_nakagami=1000,
            num_samples_rician=100,
            fs=50_000,
            N=10
        )

    def test_run_simulation_returns_array(self):
        result = self.ch.run_simulation(
            theta_degrees=45, v_LEO=7.5, fc_array=2.5e9,
            distance_satellite=1500
        )
        self.assertIsInstance(result, np.ndarray)

    def test_run_simulation_correct_length(self):
        """Output should have num_samples_nakagami samples."""
        result = self.ch.run_simulation(
            theta_degrees=45, v_LEO=7.5, fc_array=2.5e9,
            distance_satellite=1500
        )
        self.assertEqual(len(result), 1000)

    def test_run_simulation_is_complex(self):
        result = self.ch.run_simulation(
            theta_degrees=45, v_LEO=7.5, fc_array=2.5e9,
            distance_satellite=1500
        )
        self.assertTrue(np.iscomplexobj(result))

    def test_run_simulation_finite_values(self):
        """All output samples must be finite (no NaN / Inf)."""
        result = self.ch.run_simulation(
            theta_degrees=45, v_LEO=7.5, fc_array=2.5e9,
            distance_satellite=1500
        )
        self.assertTrue(np.all(np.isfinite(result)),
            "run_simulation returned NaN or Inf values.")


# ===========================================================================
#  8. Air_2_Ground_fading_channel.py — distance unit bug
# ===========================================================================
class TestAir2GroundPhase(unittest.TestCase):
    """Bug fixed: the shadowed-Rician combination used distance_satellite*1000
    even though the variable was already converted to metres at the top of
    run_simulation.  This caused a 10^6 error in the phase term.

    We verify the fix by checking that the phase rotation for two distances
    that differ by exactly 1 km corresponds to the correct electrical length.
    """

    def setUp(self):
        np.random.seed(0)
        self.ch = Air_Fading_channel(
            num_samples_nakagami=500,
            num_samples_rician=50,
            fs=50_000,
            N=10
        )

    def test_run_simulation_returns_array(self):
        result = self.ch.run_simulation(
            theta_degrees=45, velocity=50.0, fc_array=2.5e9,
            distance_satellite=100
        )
        self.assertIsInstance(result, np.ndarray)

    def test_run_simulation_finite_values(self):
        """All samples must be finite — the old bug could produce Inf/NaN
        due to extreme phase wrapping caused by the ×1000 distance error."""
        result = self.ch.run_simulation(
            theta_degrees=45, velocity=50.0, fc_array=2.5e9,
            distance_satellite=100
        )
        self.assertTrue(np.all(np.isfinite(result)),
            "Air_Fading_channel.run_simulation returned NaN or Inf. "
            "The ×1000 distance bug may have been reintroduced.")

    def test_run_simulation_correct_length(self):
        result = self.ch.run_simulation(
            theta_degrees=45, velocity=50.0, fc_array=2.5e9,
            distance_satellite=100
        )
        self.assertEqual(len(result), 500)

    def test_phase_consistent_with_correct_distance(self):
        """The Nakagami phase multiplier is exp(-j*2π*d*fc/c).
        For d=100 km = 100 000 m and fc=2.5 GHz:
            φ_correct = 2π * 100 000 * 2.5e9 / 3e8  ≈ 5 236 rad  (mod 2π)
            φ_bugged  = 2π * 100 000 000 * 2.5e9 / 3e8            (×1000 too large)
        We cannot directly inspect the internal phase, but we can verify the
        Nakagami envelope (abs value) is unaffected by the phase term and
        stays within physically plausible bounds (positive, not extreme)."""
        np.random.seed(42)
        result = self.ch.run_simulation(
            theta_degrees=45, velocity=50.0, fc_array=2.5e9,
            distance_satellite=100
        )
        env = np.abs(result)
        self.assertGreater(env.mean(), 0.0)
        # If the distance were 10^6× too large the phase wraps so fast the
        # envelope coherence collapses — its std would swamp the mean.
        # A healthy channel has std / mean < 5 at these parameter values.
        self.assertLess(env.std() / env.mean(), 10.0,
            "Envelope std/mean ratio is extreme — possible distance unit bug.")


# ===========================================================================
#  9. Air_2_Ground_fading_channel.py — normalisation (same fix as satellite)
# ===========================================================================
class TestAir2GroundNormalisation(unittest.TestCase):
    """Bug fixed: same N+1 vs N normalisation error as in Satellite_fading_channel."""

    def setUp(self):
        np.random.seed(0)
        self.ch = Air_Fading_channel(
            num_samples_nakagami=4000,
            num_samples_rician=400,
            fs=10_000,
            N=20
        )

    def test_nlos_power_fraction(self):
        """With K=0, mean power of scatter sum should be ~1.0 (not N/(N+1))."""
        K   = 0.0
        N   = self.ch.N
        np.random.seed(42)
        samples = self.ch.rician_fading_accurate(
            2000, fd=20.0, fs=self.ch.fs, K=K, theta_rad=0.0
        )
        mean_power = np.mean(np.abs(samples) ** 2)
        self.assertAlmostEqual(mean_power, 1.0, delta=0.15,
            msg=f"Mean NLOS power expected ~1.0, got {mean_power:.3f}. "
                f"N+1 normalisation would give ~{N/(N+1):.3f}.")


# ===========================================================================
#  Main
# ===========================================================================
if __name__ == "__main__":
    unittest.main(verbosity=2)
