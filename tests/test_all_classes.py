"""
test_all_classes.py
===================
Unit tests for all 3DANTS classes not covered by test_accuracy_fixes.py:

  Position and Mobility:
    - HAPS_trajectory
    - Uav_trajectory
    - terresterial_network  (no skyfield calls needed for most methods)

  Communication channel:
    - Satellite_communication_parameter
    - FadingSimulation          (fading_channel_sim.py)
    - FadingSimulation_Non_terrestrial (ntn_fading_channel_sim.py)
    - Air  (Air_objects_class.py)

  Traffic:
    - TrafficModel  (CBR, Poisson, Bursty)

Run from the repo root:
    pytest tests/test_all_classes.py -v
Or directly:
    python tests/test_all_classes.py
"""

import sys
import os
import math
import unittest
import numpy as np
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Silence heavy optional imports that are not needed for these tests
# ---------------------------------------------------------------------------
for _mod in ['skyfield', 'skyfield.api', 'sgp4', 'sgp4.api',
             'sympy', 'progress', 'progress.bar']:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(_REPO, '3DANTS', 'Position and Mobility'))
sys.path.insert(0, os.path.join(_REPO, '3DANTS', 'Communication channel'))
sys.path.insert(0, os.path.join(_REPO, '3DANTS', 'Traffic'))

from HAPS_trajec_class import HAPS_trajectory
from Uav_trajec_class import Uav_trajectory
from Terresterial_Object import terresterial_network
from Satellite_comm_param import Satellite_communication_parameter
from fading_channel_sim import FadingSimulation
from ntn_fading_channel_sim import FadingSimulation_Non_terrestrial
from Air_objects_class import Air
from Traffic_models import TrafficModel


# ===========================================================================
#  1.  HAPS_trajectory
# ===========================================================================
class TestHAPSTrajectory(unittest.TestCase):
    """HAPS_trajectory models a HAPS flying in a horizontal circle."""

    def setUp(self):
        # velocity=75 km/h, radius=6 km, time_interval=1 s
        self.haps = HAPS_trajectory(velocity=75.0, radius=6.0, time_interval=1)

    # -- angular velocity ----------------------------------------------------
    def test_angular_velocity_formula(self):
        """ω = (v / r) / 3600  [rad/s]  (velocity in km/h, radius in km)."""
        expected = (75.0 / 6.0) / 3600
        self.assertAlmostEqual(self.haps.angular_velocity, expected, places=10)

    def test_angular_velocity_positive(self):
        self.assertGreater(self.haps.angular_velocity, 0)

    # -- num_steps -----------------------------------------------------------
    def test_num_steps_one_full_circle(self):
        """num_steps should cover one full 2π revolution."""
        omega = self.haps.angular_velocity
        expected = int(2 * math.pi / (omega * self.haps.time_interval))
        self.assertEqual(self.haps.num_steps, expected)

    def test_num_steps_positive(self):
        self.assertGreater(self.haps.num_steps, 0)

    # -- get_values ----------------------------------------------------------
    def test_get_values_returns_tuple(self):
        omega, steps = self.haps.get_values()
        self.assertAlmostEqual(omega, self.haps.angular_velocity)
        self.assertEqual(steps, self.haps.num_steps)

    # -- simulate_circular_trajectory ----------------------------------------
    def test_trajectory_output_shape(self):
        """Should return a (1, 4) array: [x, y, z, elevation_angle]."""
        center = np.array([0.0, 0.0, 10.0])
        gs     = np.array([0.0, 0.0, 0.0])
        result = self.haps.simulate_circular_trajectory(center, h0=10.0, step=1, GS_position=gs)
        self.assertEqual(result.shape, (1, 4))

    def test_trajectory_z_unchanged(self):
        """HAPS flies at constant altitude — z coordinate must equal center z."""
        center = np.array([1.0, 2.0, 15.0])
        gs     = np.array([1.0, 2.0, 0.0])
        result = self.haps.simulate_circular_trajectory(center, h0=15.0, step=5, GS_position=gs)
        self.assertAlmostEqual(result[0, 2], 15.0, places=6)

    def test_trajectory_xy_on_circle(self):
        """x, y must lie on a circle of the correct radius around the center."""
        center = np.array([0.0, 0.0, 10.0])
        gs     = np.array([0.0, 0.0, 0.0])
        result = self.haps.simulate_circular_trajectory(center, h0=10.0, step=1, GS_position=gs)
        dx = result[0, 0] - center[0]
        dy = result[0, 1] - center[1]
        r  = math.sqrt(dx**2 + dy**2)
        self.assertAlmostEqual(r, self.haps.radius, places=5)

    def test_trajectory_elevation_in_range(self):
        """Elevation angle must be in [-90°, 90°]."""
        center = np.array([0.0, 0.0, 10.0])
        gs     = np.array([0.0, 0.0, 0.0])
        for step in range(1, 10):
            result = self.haps.simulate_circular_trajectory(center, h0=10.0,
                                                             step=step, GS_position=gs)
            el = result[0, 3]
            self.assertGreaterEqual(el, -90.0)
            self.assertLessEqual(el,    90.0)

    def test_different_steps_give_different_xy(self):
        """Different step values should produce different x/y positions."""
        center = np.array([0.0, 0.0, 5.0])
        gs     = np.array([0.0, 0.0, 0.0])
        pos1   = self.haps.simulate_circular_trajectory(center, h0=5.0, step=1, GS_position=gs)
        pos100 = self.haps.simulate_circular_trajectory(center, h0=5.0, step=100, GS_position=gs)
        self.assertFalse(np.allclose(pos1[0, :2], pos100[0, :2]),
                         "Steps 1 and 100 should give different xy positions.")


# ===========================================================================
#  2.  Uav_trajectory
# ===========================================================================
class TestUavTrajectory(unittest.TestCase):
    """Uav_trajectory is structurally identical to HAPS_trajectory but uses
    h0 (height above GS) for the elevation calculation instead of z0 - GS[2]."""

    def setUp(self):
        self.uav = Uav_trajectory(velocity=18.0, radius=2.0, time_interval=1)

    def test_angular_velocity_formula(self):
        expected = (18.0 / 2.0) / 3600
        self.assertAlmostEqual(self.uav.angular_velocity, expected, places=10)

    def test_num_steps_full_circle(self):
        omega    = self.uav.angular_velocity
        expected = int(2 * math.pi / (omega * self.uav.time_interval))
        self.assertEqual(self.uav.num_steps, expected)

    def test_trajectory_shape(self):
        center = np.array([0.0, 0.0, 0.5])
        gs     = np.array([0.0, 0.0, 0.0])
        result = self.uav.simulate_circular_trajectory(center, h0=0.5, step=1, GS_position=gs)
        self.assertEqual(result.shape, (1, 4))

    def test_trajectory_z_constant(self):
        center = np.array([5.0, 5.0, 2.0])
        gs     = np.array([5.0, 5.0, 0.0])
        for step in [1, 50, 200]:
            result = self.uav.simulate_circular_trajectory(center, h0=2.0, step=step, GS_position=gs)
            self.assertAlmostEqual(result[0, 2], 2.0, places=6)

    def test_trajectory_radius(self):
        """x,y must lie on circle of correct radius around center."""
        center = np.array([0.0, 0.0, 1.0])
        gs     = np.array([0.0, 0.0, 0.0])
        result = self.uav.simulate_circular_trajectory(center, h0=1.0, step=10, GS_position=gs)
        dx = result[0, 0] - center[0]
        dy = result[0, 1] - center[1]
        self.assertAlmostEqual(math.sqrt(dx**2 + dy**2), self.uav.radius, places=5)

    def test_elevation_bounded(self):
        center = np.array([0.0, 0.0, 1.0])
        gs     = np.array([0.0, 0.0, 0.0])
        for step in range(1, 20):
            r  = self.uav.simulate_circular_trajectory(center, h0=1.0, step=step, GS_position=gs)
            el = r[0, 3]
            self.assertGreaterEqual(el, -90.0)
            self.assertLessEqual(el,    90.0)

    def test_uav_cim_trajectory_shape(self):
        """UAV_trajectory_CIM must return a (1, 4) array."""
        center = np.array([0.0, 0.0, 1.0])
        ref    = np.array([0.0, 0.0, 0.0])
        result = self.uav.UAV_trajectory_CIM(center, speed=30.0, time_step=1.0,
                                              reference_point=ref, GS_position=ref)
        self.assertEqual(result.shape, (1, 4))

    def test_uav_cim_z_above_gs(self):
        """CIM trajectory z must remain above the GS z level."""
        gs     = np.array([0.0, 0.0, 0.0])
        center = np.array([0.0, 0.0, 0.5])
        result = self.uav.UAV_trajectory_CIM(center, speed=10.0, time_step=1.0,
                                              reference_point=gs, GS_position=gs)
        self.assertGreater(result[0, 2], gs[2])


# ===========================================================================
#  3.  terresterial_network
# ===========================================================================
class TestTerrestrialNetwork(unittest.TestCase):

    def setUp(self):
        self.net = terresterial_network(fc=2.5e9, environment='Suburban')

    # -- frequency conversion ------------------------------------------------
    def test_fc_stored_in_ghz(self):
        """fc is stored in GHz internally (divided by 1e9)."""
        self.assertAlmostEqual(self.net.fc, 2.5)

    # -- UE PPP generation ---------------------------------------------------
    def test_ue_generation_shape(self):
        """PPP generation must return (num_points, 3) array."""
        center = np.array([0.0, 0.0, 6371.0])
        pos    = self.net.UE_random_PPP_generation(center, radius=1.0, num_points=50)
        self.assertEqual(pos.shape, (50, 3))

    def test_ue_generation_within_radius(self):
        """All UE x,y coordinates must fall within the requested radius."""
        center = np.array([0.0, 0.0, 6371.0])
        np.random.seed(0)
        pos = self.net.UE_random_PPP_generation(center, radius=2.0, num_points=200)
        dist_2d = np.sqrt((pos[:, 0] - center[0])**2 + (pos[:, 1] - center[1])**2)
        self.assertTrue(np.all(dist_2d <= 2.0 + 1e-9),
                        "Some UEs generated outside the requested radius.")

    def test_ue_z_offset(self):
        """UE z is set to center[2] - 0.034 (ground-level offset in km)."""
        center = np.array([0.0, 0.0, 6371.0])
        pos    = self.net.UE_random_PPP_generation(center, radius=1.0, num_points=10)
        expected_z = center[2] - 0.034
        self.assertTrue(np.allclose(pos[:, 2], expected_z))

    # -- UAV cone PPP generation ---------------------------------------------
    def test_uav_cone_shape(self):
        center = np.array([0.0, 0.0, 0.0])
        pos    = self.net.UAV_PPP_generation_inisde_a_cone(center, r=2.0, h=1.0, numbPoints=30)
        self.assertEqual(pos.shape[1], 3)
        self.assertEqual(pos.shape[0], 30)

    def test_uav_cone_z_bounds(self):
        """All UAV z values should be within [center_z, center_z + h]."""
        center = np.array([0.0, 0.0, 0.5])
        np.random.seed(1)
        pos = self.net.UAV_PPP_generation_inisde_a_cone(center, r=3.0, h=2.0, numbPoints=100)
        self.assertTrue(np.all(pos[:, 2] >= center[2] - 1e-9))
        self.assertTrue(np.all(pos[:, 2] <= center[2] + 2.0 + 1e-9))

    # -- base station generation ---------------------------------------------
    def test_bs_generation_count(self):
        np.random.seed(42)
        bs = self.net.generate_base_station_positions(
            center_x=0.0, center_y=0.0, circle_radius=10.0,
            num_base_stations=5, fixed_z=0.035, r_bs=0.5
        )
        self.assertEqual(len(bs), 5)

    def test_bs_separation(self):
        """All pairs of BSs must be at least 2*r_bs apart."""
        np.random.seed(7)
        r_bs = 1.0
        bs   = self.net.generate_base_station_positions(
            0.0, 0.0, 20.0, 6, 0.035, r_bs
        )
        for i in range(len(bs)):
            for j in range(i + 1, len(bs)):
                d = math.hypot(bs[i][0] - bs[j][0], bs[i][1] - bs[j][1])
                self.assertGreaterEqual(d, 2 * r_bs - 1e-9,
                    f"BS pair ({i},{j}) too close: {d:.4f} < {2*r_bs}")

    def test_bs_within_circle(self):
        """All generated BSs must lie inside the requested circle radius."""
        np.random.seed(3)
        R  = 15.0
        bs = self.net.generate_base_station_positions(0.0, 0.0, R, 4, 0.035, 0.5)
        for x, y, _ in bs:
            self.assertLessEqual(math.hypot(x, y), R + 1e-9)

    # -- LoS probability (Pr_LOS_RMa) ----------------------------------------
    def test_pr_los_short_distance_is_one(self):
        """d_2D ≤ 10 m → LoS probability should be 1 for all environments."""
        for env in ['Suburban', 'Urban']:
            net = terresterial_network(fc=2.5e9, environment=env)
            p   = net.Pr_LOS_RMa(d_2D_out=5.0, h_UT=1.5)
            self.assertAlmostEqual(p, 1.0, places=6,
                msg=f"Pr_LOS({env}, 5m) expected 1.0, got {p:.4f}")

    def test_pr_los_decreases_with_distance(self):
        """LoS probability should decrease as distance increases."""
        net = terresterial_network(fc=2.5e9, environment='Urban')
        p_near = net.Pr_LOS_RMa(100.0, 1.5)
        p_far  = net.Pr_LOS_RMa(500.0, 1.5)
        self.assertGreater(p_near, p_far)

    def test_pr_los_in_valid_range(self):
        """LoS probability must be in [0, 1]."""
        net = terresterial_network(fc=2.5e9, environment='Urban')
        for d in [10, 50, 100, 500, 1000, 5000]:
            p = net.Pr_LOS_RMa(float(d), 1.5)
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0 + 1e-9)

    # -- convert_to_nearest --------------------------------------------------
    def test_convert_to_nearest_multiples_of_10(self):
        net = terresterial_network(fc=2.5e9, environment='Suburban')
        self.assertEqual(net.convert_to_nearest(45), 40)  # tie → smaller decade
        self.assertEqual(net.convert_to_nearest(44), 40)
        self.assertEqual(net.convert_to_nearest(30), 30)
        self.assertEqual(net.convert_to_nearest(35), 30)  # tie → smaller decade

    # -- doppler calculation -------------------------------------------------
    def test_doppler_shape(self):
        """doppler_Sat_UE must return (num_sats, num_angles) array."""
        net = terresterial_network(fc=2.5e9, environment='Suburban')
        vel_Sat_UEs = np.array([[7500.0], [7500.0]])    # 2 satellites, 1 UE each
        alpha_n     = np.linspace(0, 2 * np.pi, 8)
        doppler     = net.doppler_Sat_UE(vel_Sat_UEs, alpha_n)
        self.assertEqual(doppler.shape, (2, 8))

    def test_doppler_zero_for_orthogonal_angle(self):
        """At α = π/2, cos(α) = 0 → Doppler shift should be zero."""
        net = terresterial_network(fc=2.5e9, environment='Suburban')
        vel = np.array([[1000.0]])
        d   = net.doppler_Sat_UE(vel, np.array([math.pi / 2]))
        self.assertAlmostEqual(d[0, 0], 0.0, places=6)

    # -- cartesian_to_latlon -------------------------------------------------
    def test_cartesian_to_latlon_north_pole(self):
        """A point at [0, 0, R] should be at lat=90°, lon=0°."""
        net   = terresterial_network(fc=2.5e9, environment='Suburban')
        R     = 6371.0
        latlons = net.cartesian_to_latlon([(0.0, 0.0, R)])
        lat, lon = latlons[0]
        self.assertAlmostEqual(lat, 90.0, delta=0.01)

    def test_cartesian_to_latlon_equator(self):
        """A point at [R, 0, 0] should be at lat≈0°, lon≈0°."""
        net = terresterial_network(fc=2.5e9, environment='Suburban')
        latlons = net.cartesian_to_latlon([(6371.0, 0.0, 0.0)])
        lat, lon = latlons[0]
        self.assertAlmostEqual(lat, 0.0, delta=0.01)
        self.assertAlmostEqual(lon, 0.0, delta=0.01)


# ===========================================================================
#  4.  Satellite_communication_parameter
# ===========================================================================
class TestSatelliteCommParam(unittest.TestCase):

    def setUp(self):
        self.scp = Satellite_communication_parameter()

    # -- S-band DL -----------------------------------------------------------
    def test_s_band_dl_returns_five_values(self):
        result = self.scp.parameters(2.5e9, 'S', 'DL')
        self.assertEqual(len(result), 5)

    def test_s_band_dl_eirp_density(self):
        eirp, *_ = self.scp.parameters(2.5e9, 'S', 'DL')
        self.assertEqual(eirp, 34)

    def test_s_band_dl_max_gain(self):
        _, gain, *_ = self.scp.parameters(2.5e9, 'S', 'DL')
        self.assertEqual(gain, 30)

    def test_s_band_dl_bandwidth(self):
        *_, bw = self.scp.parameters(2.5e9, 'S', 'DL')
        self.assertEqual(bw, 30)

    # -- Ka-band DL ----------------------------------------------------------
    def test_ka_band_dl_returns_five_values(self):
        result = self.scp.parameters(20e9, 'Ka', 'DL')
        self.assertEqual(len(result), 5)

    def test_ka_band_dl_higher_eirp_than_s(self):
        eirp_s,  *_ = self.scp.parameters(2.5e9,  'S',  'DL')
        eirp_ka, *_ = self.scp.parameters(20e9,   'Ka', 'DL')
        self.assertGreater(eirp_ka, eirp_s,
            "Ka-band EIRP density should be higher than S-band.")

    def test_ka_band_dl_narrower_beamwidth(self):
        """Ka-band has smaller beam → narrower 3 dB beamwidth than S-band."""
        _, _, bw_s,  *_ = self.scp.parameters(2.5e9,  'S',  'DL')
        _, _, bw_ka, *_ = self.scp.parameters(20e9,   'Ka', 'DL')
        self.assertLess(bw_ka, bw_s)

    def test_ka_band_dl_larger_bandwidth(self):
        *_, bw_s  = self.scp.parameters(2.5e9,  'S',  'DL')
        *_, bw_ka = self.scp.parameters(20e9,   'Ka', 'DL')
        self.assertGreater(bw_ka, bw_s)

    # -- S-band UL -----------------------------------------------------------
    def test_s_band_ul_returns_three_values(self):
        result = self.scp.parameters(2.5e9, 'S', 'UL')
        self.assertEqual(len(result), 3)

    def test_s_band_ul_g_over_t(self):
        g_over_t, _, _ = self.scp.parameters(2.5e9, 'S', 'UL')
        self.assertEqual(g_over_t, 1.1)

    # -- Ka-band UL ----------------------------------------------------------
    def test_ka_band_ul_higher_g_over_t(self):
        got_s,  *_ = self.scp.parameters(2.5e9,  'S',  'UL')
        got_ka, *_ = self.scp.parameters(20e9,   'Ka', 'UL')
        self.assertGreater(got_ka, got_s,
            "Ka-band G/T should be higher than S-band.")

    # -- physical reasonableness ---------------------------------------------
    def test_dl_gain_values_in_reasonable_range(self):
        """Satellite Tx gain should be between 20 and 60 dBi."""
        for band in ['S', 'Ka']:
            _, gain, *_ = self.scp.parameters(2.5e9, band, 'DL')
            self.assertGreater(gain, 20)
            self.assertLess(gain, 60)


# ===========================================================================
#  5.  FadingSimulation  (fading_channel_sim.py)
# ===========================================================================
class TestFadingSimulation(unittest.TestCase):
    """Tests for the original FadingSimulation class."""

    def setUp(self):
        np.random.seed(0)
        # Default: full mode (Option C) — true orbital velocity
        self.sim = FadingSimulation(
            num_samples=1000, fs=10_000, K=5.0,
            N=16, h=600_000.0, doppler_mode='full'
        )

    # -- satellite velocity and Doppler modes --------------------------------
    def test_full_mode_uses_orbital_velocity(self):
        """Option C (full): v_sat must equal vis-viva sqrt(GM/r)."""
        G = 6.6743e-11; M = 5.9722e24; R = 6371e3; h = 600_000.0
        expected = math.sqrt(G * M / (R + h))
        sim = FadingSimulation(100, 1000, 5.0, 16, h, doppler_mode='full')
        self.assertAlmostEqual(sim.v_sat, expected, delta=1.0)

    def test_compensated_mode_uses_ue_speed(self):
        """Option A (compensated): v_sat must equal the supplied UE speed."""
        v_ue = 3.5  # m/s
        sim  = FadingSimulation(100, 1000, 5.0, 16, 600_000.0,
                                doppler_mode='compensated', v_residual_mps=v_ue)
        self.assertAlmostEqual(sim.v_sat, v_ue, places=9)

    def test_scaled_mode_hits_target_coherence_time(self):
        """Option B (scaled): Tc_effective must equal the requested Tc_target_s."""
        Tc_target = 0.05   # 50 ms
        sim = FadingSimulation(100, 1000, 5.0, 16, 600_000.0,
                               doppler_mode='scaled', Tc_target_s=Tc_target, fc_ref=2.5e9)
        self.assertAlmostEqual(sim.Tc_effective, Tc_target, delta=1e-9)

    def test_legacy_no_maps_to_full_mode(self):
        """Backward-compat: Doppler_compensate='No' → full mode, orbital velocity."""
        G = 6.6743e-11; M = 5.9722e24; R = 6371e3; h = 600_000.0
        expected = math.sqrt(G * M / (R + h))
        sim = FadingSimulation(100, 1000, 5.0, 16, h, Doppler_compensate='No')
        self.assertAlmostEqual(sim.v_sat, expected, delta=1.0)

    def test_legacy_yes_maps_to_compensated_mode(self):
        """Backward-compat: Doppler_compensate='Yes' → compensated mode,
        v_sat = 1.5 m/s (pedestrian default), NOT the old sqrt(GM/r^2) error."""
        sim = FadingSimulation(100, 1000, 5.0, 16, 600_000.0, Doppler_compensate='Yes')
        self.assertEqual(sim.doppler_mode, 'compensated')
        self.assertAlmostEqual(sim.v_sat, 1.5, places=9)

    def test_full_mode_velocity_much_larger_than_compensated(self):
        """Orbital speed (7562 m/s) >> UE residual (1.5 m/s): >1000x difference."""
        sim_full = FadingSimulation(100, 1000, 5.0, 16, 600_000.0, doppler_mode='full')
        sim_comp = FadingSimulation(100, 1000, 5.0, 16, 600_000.0,
                                   doppler_mode='compensated', v_residual_mps=1.5)
        self.assertGreater(sim_full.v_sat / sim_comp.v_sat, 1000)

    def test_satellite_velocity_positive(self):
        self.assertGreater(self.sim.v_sat, 0)

    # -- Doppler frequency ---------------------------------------------------
    def test_f_D_at_zero_elevation_is_max(self):
        """Full mode: at θ=0 (head-on) Doppler is maximum."""
        fd_0   = self.sim.f_D(theta=0.0,         fc=2.5e9)
        fd_pi4 = self.sim.f_D(theta=math.pi / 4, fc=2.5e9)
        self.assertGreater(fd_0, fd_pi4)

    def test_f_D_at_90_degrees_is_zero(self):
        """Full mode: at θ=π/2 (broadside) Doppler is zero."""
        fd = self.sim.f_D(theta=math.pi / 2, fc=2.5e9)
        self.assertAlmostEqual(fd, 0.0, places=3)

    def test_f_D_positive_frequency(self):
        for theta in [0.0, 0.3, 0.6, 1.0]:
            self.assertGreaterEqual(self.sim.f_D(theta=theta, fc=2.5e9), 0.0)

    def test_compensated_mode_f_D_independent_of_theta(self):
        """Option A/B: f_D does not depend on theta (UE direction is random)."""
        sim = FadingSimulation(100, 1000, 5.0, 16, 600_000.0,
                               doppler_mode='compensated', v_residual_mps=1.5)
        fd_0   = sim.f_D(theta=0.0,         fc=2.5e9)
        fd_pi4 = sim.f_D(theta=math.pi / 4, fc=2.5e9)
        fd_pi2 = sim.f_D(theta=math.pi / 2, fc=2.5e9)
        self.assertAlmostEqual(fd_0, fd_pi4, places=9)
        self.assertAlmostEqual(fd_0, fd_pi2, places=9)

    def test_scaled_mode_f_D_matches_target_tc(self):
        """Option B: f_D should equal 0.423/Tc_target at fc_ref."""
        Tc = 0.025  # 25 ms
        fc = 2.5e9
        sim = FadingSimulation(100, 1000, 5.0, 16, 600_000.0,
                               doppler_mode='scaled', Tc_target_s=Tc, fc_ref=fc)
        expected_fd = 0.423 / Tc
        self.assertAlmostEqual(sim.f_D(0, fc), expected_fd, delta=0.01)

    # -- rician_fading_accurate ----------------------------------------------
    def test_rician_output_length(self):
        fd      = self.sim.f_D(math.radians(45), 2.5e9)
        samples = self.sim.rician_fading_accurate(fd)
        self.assertEqual(len(samples), self.sim.num_samples)

    def test_rician_output_complex(self):
        fd      = self.sim.f_D(math.radians(45), 2.5e9)
        samples = self.sim.rician_fading_accurate(fd)
        self.assertTrue(np.iscomplexobj(samples))

    def test_rician_output_finite(self):
        fd      = self.sim.f_D(math.radians(45), 2.5e9)
        samples = self.sim.rician_fading_accurate(fd)
        self.assertTrue(np.all(np.isfinite(samples)))

    # -- nakagami_m_based_Gamma ----------------------------------------------
    def test_nakagami_positive_values(self):
        samples = self.sim.nakagami_m_based_Gamma(m=2.0, Omega=1.0)
        self.assertTrue(np.all(samples > 0))

    def test_nakagami_length(self):
        samples = self.sim.nakagami_m_based_Gamma(m=2.0, Omega=1.0)
        self.assertEqual(len(samples), self.sim.num_samples)

    def test_nakagami_mean_power_scales_with_omega(self):
        """E[X²] ≈ Omega for large N."""
        np.random.seed(99)
        s1 = self.sim.nakagami_m_based_Gamma(m=5.0, Omega=1.0)
        s2 = self.sim.nakagami_m_based_Gamma(m=5.0, Omega=4.0)
        ratio = np.mean(s2**2) / np.mean(s1**2)
        self.assertAlmostEqual(ratio, 4.0, delta=0.6)

    # -- Rank_matching -------------------------------------------------------
    def test_rank_matching_preserves_length(self):
        np.random.seed(5)
        r = np.random.randn(100)
        n = np.abs(np.random.randn(100))
        matched = self.sim.Rank_matching(r, n)
        self.assertEqual(len(matched), 100)

    def test_rank_matching_preserves_sorted_order(self):
        """After rank-matching, the sorted matched array should equal sorted nakagami."""
        np.random.seed(6)
        r = np.random.randn(200)
        n = np.abs(np.random.randn(200))
        matched = self.sim.Rank_matching(r, n)
        self.assertTrue(np.allclose(np.sort(matched), np.sort(n)))

    # -- channel parameter polynomials ---------------------------------------
    def test_b0_positive_range(self):
        for theta in range(10, 91, 10):
            self.assertGreater(self.sim.b_0_calc(theta), 0.01)

    def test_m_theta_positive(self):
        for theta in range(10, 91, 10):
            self.assertGreater(self.sim.m_theta(theta), 0)

    def test_omega_theta_positive_for_valid_angles(self):
        """Omega_theta is the polynomial fit from Lutz/3GPP. It can be negative
        at very low elevation (≤10°) — run_simulation uses a fallback in that
        case. Test that it is positive for angles ≥ 20°."""
        for theta in range(20, 91, 10):
            self.assertGreater(self.sim.Omega_theta(theta), 0,
                msg=f"Omega_theta({theta}°) should be positive")


# ===========================================================================
#  6.  FadingSimulation_Non_terrestrial  (ntn_fading_channel_sim.py)
# ===========================================================================
class TestFadingSimulationNTN(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.sim = FadingSimulation_Non_terrestrial(
            num_samples=500, fs=10_000, N=12,
            h=600_000.0, doppler_mode='full'
        )

    def test_velocity_positive(self):
        self.assertGreater(self.sim.v_sat, 0)

    def test_f_D_broadside_zero(self):
        self.assertAlmostEqual(self.sim.f_D(math.pi / 2, 2.5e9), 0.0, places=3)

    def test_run_simulation_length(self):
        result = self.sim.run_simulation(45, 2.5e9, 1500)
        self.assertEqual(len(result), self.sim.num_samples)

    def test_run_simulation_complex(self):
        result = self.sim.run_simulation(45, 2.5e9, 1500)
        self.assertTrue(np.iscomplexobj(result))

    def test_run_simulation_finite(self):
        result = self.sim.run_simulation(45, 2.5e9, 1500)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_run_simulation_multiple_elevations(self):
        """Should work for a list of elevation angles."""
        result = self.sim.run_simulation([30, 45, 60], 2.5e9, 1500)
        self.assertEqual(len(result), self.sim.num_samples)

    def test_channel_with_ue_movement_length(self):
        result = self.sim.channel_with_UE_movement(
            theta_degrees=45, fc_array=2.5e9,
            UE_speed=1.0, UE_moving_direction=0.0,
            distance_satellite=1500
        )
        self.assertEqual(len(result), self.sim.num_samples)

    def test_channel_with_ue_movement_finite(self):
        result = self.sim.channel_with_UE_movement(45, 2.5e9, 1.0, 0.0, 1500)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_nakagami_fallback_on_negative_m(self):
        """m_theta can return negative for very low elevation.
        run_simulation should handle this by using fallback m=0.835."""
        result = self.sim.run_simulation(theta_degrees=5, fc_array=2.5e9,
                                          distance_satellite=1500)
        self.assertTrue(np.all(np.isfinite(result)),
            "run_simulation should not produce NaN for low elevation angles.")

    def test_rank_matching_preserves_distribution(self):
        np.random.seed(10)
        r       = np.abs(np.random.randn(500))
        n       = np.abs(np.random.randn(500))
        matched = self.sim.Rank_matching(r, n)
        self.assertTrue(np.allclose(np.sort(matched), np.sort(n)))


# ===========================================================================
#  7.  Air  (Air_objects_class.py)
# ===========================================================================
class TestAir(unittest.TestCase):

    # -- environment parameters ----------------------------------------------
    def test_suburban_params(self):
        air = Air('Suburban', 2.5e9)
        self.assertEqual(air.alpha, 0.1)
        self.assertEqual(air.betta, 750)
        self.assertEqual(air.gamma, 8)

    def test_urban_params(self):
        air = Air('Urban', 2.5e9)
        self.assertEqual(air.alpha, 0.3)
        self.assertEqual(air.betta, 500)
        self.assertEqual(air.gamma, 15)

    def test_dense_urban_params(self):
        air = Air('DenseUrban', 2.5e9)
        self.assertEqual(air.alpha, 0.5)
        self.assertEqual(air.betta, 300)
        self.assertEqual(air.gamma, 20)

    # -- LoS calculator ------------------------------------------------------
    def test_los_probability_in_range(self):
        """LoS probability must be in [0, 1] for all environments and angles."""
        for env in ['Suburban', 'Urban', 'DenseUrban']:
            air = Air(env, 2.5e9)
            for angle in [10, 30, 45, 60, 90]:
                p = air.LoS_calculator(angle)
                self.assertGreaterEqual(p, 0.0,
                    f"LoS prob negative for {env} at {angle}°")
                self.assertLessEqual(p, 1.0 + 1e-9,
                    f"LoS prob > 1 for {env} at {angle}°")

    def test_los_increases_with_elevation(self):
        """LoS probability should be higher at 45° than at 10° for Urban/DenseUrban.
        Suburban is near-saturated at both angles so we only check the trend
        for Urban and DenseUrban where the gradient is clearly positive."""
        for env in ['Urban', 'DenseUrban']:
            air  = Air(env, 2.5e9)
            p_lo = air.LoS_calculator(10)
            p_hi = air.LoS_calculator(45)
            self.assertGreaterEqual(p_hi, p_lo,
                f"LoS prob should increase with elevation in {env} env.")

    def test_los_suburban_higher_than_dense_urban(self):
        """The Al-Hourani LoS sigmoid uses different alpha/beta/gamma per environment.
        DenseUrban (alpha=0.5) has a denser building profile than Suburban (alpha=0.1).
        At moderate-to-high elevation (60°), Suburban should produce a LoS probability
        at least as high as DenseUrban. Note: the empirical polynomial fit can slightly
        exceed 1.0 at some angles — we use assertGreaterEqual with a tolerance."""
        sub = float(Air('Suburban',   2.5e9).LoS_calculator(60))
        den = float(Air('DenseUrban', 2.5e9).LoS_calculator(60))
        self.assertGreaterEqual(sub, den - 0.01,
            f"Suburban LoS({sub:.4f}) should be ≥ DenseUrban LoS({den:.4f}) at 60°")

    # -- general path loss ---------------------------------------------------
    def test_general_pathloss_returns_two_values(self):
        air      = Air('Urban', 2.5e9)
        gs       = np.array([[0.0, 0.0, 0.0]])
        uav      = np.array([[0.0, 0.0, 1.0]])
        pl_los, pl_nlos = air.general_pathloss_calculator(gs, uav)
        self.assertIsInstance(pl_los,  float)
        self.assertIsInstance(pl_nlos, float)

    def test_general_pathloss_nlos_greater_than_los(self):
        """NLoS path loss should always exceed LoS path loss."""
        for env in ['Suburban', 'Urban']:
            air  = Air(env, 2.5e9)
            gs   = np.array([[0.0, 0.0, 0.0]])
            uav  = np.array([[0.0, 0.0, 2.0]])
            lo, nl = air.general_pathloss_calculator(gs, uav)
            self.assertGreater(nl, lo,
                f"NLoS PL should exceed LoS PL in {env} environment.")

    def test_general_pathloss_increases_with_distance(self):
        """Path loss must increase as distance increases."""
        air  = Air('Urban', 2.5e9)
        gs   = np.array([[0.0, 0.0, 0.0]])
        near = np.array([[0.0, 0.0, 1.0]])
        far  = np.array([[0.0, 0.0, 5.0]])
        pl_near, _ = air.general_pathloss_calculator(gs, near)
        pl_far,  _ = air.general_pathloss_calculator(gs, far)
        self.assertGreater(pl_far, pl_near)

    # -- air-to-air K factor -------------------------------------------------
    def test_air2air_k_positive(self):
        """Air-to-air K factor must be positive."""
        air = Air('Urban', 2.5e9)
        K   = air.air2air_K_calculator(h1=100.0, h2=200.0, rho_direct=0.5)
        self.assertGreater(K, 0)

    def test_air2air_k_increases_with_rho(self):
        """Stronger direct component (larger ρ) → higher K."""
        air = Air('Urban', 2.5e9)
        K1  = air.air2air_K_calculator(100.0, 200.0, rho_direct=0.5)
        K2  = air.air2air_K_calculator(100.0, 200.0, rho_direct=2.0)
        self.assertGreater(K2, K1)

    # -- air-to-air path loss ------------------------------------------------
    def test_air2air_pathloss_positive(self):
        """CI path loss should be a positive number."""
        air = Air('Urban', 2.5e9)
        pos1 = np.array([0.0, 0.0, 1.0])
        pos2 = np.array([0.0, 0.5, 1.0])
        pl   = air.Air2Air_pathloss(pos1, pos2)
        self.assertGreater(float(pl), 0.0)

    def test_air2air_pathloss_increases_with_distance(self):
        air  = Air('Urban', 2.5e9)
        pos1 = np.array([0.0, 0.0, 1.0])
        near = np.array([0.1, 0.0, 1.0])
        far  = np.array([5.0, 0.0, 1.0])
        pl_n = air.Air2Air_pathloss(pos1, near)
        pl_f = air.Air2Air_pathloss(pos1, far)
        self.assertGreater(float(pl_f), float(pl_n))

    # -- Rician K from elevation angle ---------------------------------------
    def test_rician_k_factor_positive(self):
        air = Air('Urban', 2.5e9)
        K   = air.Rician_factor_calculator(elevation_angle=45, K0=-10, K_pi_half=10)
        self.assertGreater(float(K.flat[0]), 0)

    def test_rician_k_increases_with_elevation(self):
        """Higher elevation angle → higher K (stronger LoS)."""
        air = Air('Urban', 2.5e9)
        K_lo = air.Rician_factor_calculator(10,  K0=-10, K_pi_half=10)
        K_hi = air.Rician_factor_calculator(80,  K0=-10, K_pi_half=10)
        self.assertGreater(float(K_hi.flat[0]), float(K_lo.flat[0]))

    # -- convert_to_nearest --------------------------------------------------
    def test_convert_to_nearest(self):
        air = Air('Urban', 2.5e9)
        self.assertEqual(air.convert_to_nearest(33), 30)
        self.assertEqual(air.convert_to_nearest(35), 30)  # tie → smaller decade
        self.assertEqual(air.convert_to_nearest(89), 90)

    # -- assign_uavs_to_bs ---------------------------------------------------
    def test_assign_uavs_shape(self):
        """Output shape should match input UAV positions shape."""
        air = Air('Urban', 2.5e9)
        uav_pos = np.random.rand(2, 5, 3)  # 2 BSs, 5 UAVs, 3D
        bs_pos  = np.random.rand(2, 3)     # 2 BSs, 3D
        result  = air.assign_uavs_to_bs(uav_pos, bs_pos)
        self.assertEqual(result.shape, uav_pos.shape)


# ===========================================================================
#  8.  TrafficModel
# ===========================================================================
class TestTrafficModel(unittest.TestCase):

    # -- initialisation ------------------------------------------------------
    def test_invalid_model_type_raises(self):
        with self.assertRaises(ValueError):
            TrafficModel('invalid_type')

    def test_cbr_defaults(self):
        tm = TrafficModel('CBR')
        self.assertEqual(tm.packet_size, 1000)
        self.assertEqual(tm.packet_rate, 10)

    def test_poisson_defaults(self):
        tm = TrafficModel('Poisson')
        self.assertEqual(tm.avg_packet_rate, 50)

    def test_bursty_defaults(self):
        tm = TrafficModel('Bursty')
        self.assertEqual(tm.on_duration,  1.0)
        self.assertEqual(tm.off_duration, 2.0)

    def test_case_insensitive(self):
        """Model type matching should be case-insensitive."""
        for variant in ['cbr', 'CBR', 'Cbr']:
            tm = TrafficModel(variant)
            self.assertEqual(tm.model_type, 'cbr')

    # -- CBR -----------------------------------------------------------------
    def test_cbr_packet_count(self):
        """CBR with rate 10 pkt/s over 1 second should produce ~10 packets."""
        tm = TrafficModel('CBR', packet_rate=10, packet_size=1000)
        tm.generate_packets(1.0)
        self.assertAlmostEqual(len(tm.packet_log), 10, delta=1)

    def test_cbr_packet_sizes_constant(self):
        tm = TrafficModel('CBR', packet_rate=10, packet_size=512)
        tm.generate_packets(1.0)
        sizes = [s for _, s in tm.packet_log]
        self.assertTrue(all(s == 512 for s in sizes))

    def test_cbr_inter_packet_time(self):
        """CBR inter-packet time must be 1/rate."""
        tm = TrafficModel('CBR', packet_rate=5)
        tm.generate_packets(2.0)
        times = [t for t, _ in tm.packet_log]
        if len(times) >= 2:
            gaps = [times[i+1] - times[i] for i in range(len(times)-1)]
            for g in gaps:
                self.assertAlmostEqual(g, 1.0 / 5, places=9)

    def test_cbr_packets_within_duration(self):
        """No packet should be generated after the requested duration."""
        tm = TrafficModel('CBR', packet_rate=20)
        duration = 2.0
        tm.generate_packets(duration)
        times = [t for t, _ in tm.packet_log]
        self.assertTrue(all(t < duration for t in times))

    def test_cbr_scales_with_duration(self):
        """Longer duration should produce proportionally more packets.
        Use a high rate so boundary effects are negligible."""
        tm = TrafficModel('CBR', packet_rate=100)  # inter_time = 0.01 s
        tm.generate_packets(1.0); n1 = len(tm.packet_log)
        tm.generate_packets(2.0); n2 = len(tm.packet_log)
        # With 100 pkt/s: n1≈100, n2≈200, ratio≈2.0 ± 1%
        self.assertAlmostEqual(n2 / n1, 2.0, delta=0.02)

    # -- Poisson -------------------------------------------------------------
    def test_poisson_generates_packets(self):
        np.random.seed(0)
        tm = TrafficModel('Poisson', avg_packet_rate=50)
        tm.generate_packets(1.0)
        self.assertGreater(len(tm.packet_log), 0)

    def test_poisson_avg_rate(self):
        """Mean packet count over 1 s should be close to avg_packet_rate."""
        np.random.seed(42)
        rate = 50
        tm   = TrafficModel('Poisson', avg_packet_rate=rate)
        tm.generate_packets(1.0)
        # Allow ±40% tolerance (statistical)
        self.assertAlmostEqual(len(tm.packet_log), rate, delta=rate * 0.4)

    def test_poisson_min_packet_size(self):
        """Poisson packets must be at least 100 bytes (enforced by max(100, ...))."""
        np.random.seed(0)
        tm = TrafficModel('Poisson', avg_packet_rate=50,
                          packet_size_mean=100, packet_size_std=500)
        tm.generate_packets(2.0)
        sizes = [s for _, s in tm.packet_log]
        self.assertTrue(all(s >= 100 for s in sizes))

    def test_poisson_within_duration(self):
        np.random.seed(0)
        tm = TrafficModel('Poisson', avg_packet_rate=20)
        tm.generate_packets(3.0)
        self.assertTrue(all(t < 3.0 for t, _ in tm.packet_log))

    # -- Bursty --------------------------------------------------------------
    def test_bursty_generates_packets(self):
        tm = TrafficModel('Bursty', on_duration=1.0, off_duration=1.0,
                          packet_rate_on=20, packet_size=500)
        tm.generate_packets(5.0)
        self.assertGreater(len(tm.packet_log), 0)

    def test_bursty_packets_within_duration(self):
        tm = TrafficModel('Bursty', on_duration=0.5, off_duration=0.5,
                          packet_rate_on=10)
        tm.generate_packets(4.0)
        self.assertTrue(all(t < 4.0 for t, _ in tm.packet_log))

    def test_bursty_fewer_than_continuous(self):
        """Bursty traffic (50% duty cycle) should produce fewer packets than CBR."""
        np.random.seed(0)
        duration = 10.0
        cbr = TrafficModel('CBR', packet_rate=20)
        cbr.generate_packets(duration)

        bursty = TrafficModel('Bursty', on_duration=1.0, off_duration=1.0,
                               packet_rate_on=20, packet_size=1000)
        bursty.generate_packets(duration)

        self.assertLess(len(bursty.packet_log), len(cbr.packet_log))

    def test_bursty_packet_sizes_constant(self):
        tm = TrafficModel('Bursty', packet_size=256, packet_rate_on=10)
        tm.generate_packets(3.0)
        sizes = [s for _, s in tm.packet_log]
        self.assertTrue(all(s == 256 for s in sizes))

    # -- get_packets_at_time -------------------------------------------------
    def test_get_packets_at_time_returns_list(self):
        tm = TrafficModel('CBR', packet_rate=10)
        tm.generate_packets(1.0)
        result = tm.get_packets_at_time(current_time_ms=100)
        self.assertIsInstance(result, list)

    def test_get_packets_at_time_correct_window(self):
        """Packets returned must fall within the requested 1 ms window."""
        tm = TrafficModel('CBR', packet_rate=1000)  # 1 pkt/ms
        tm.generate_packets(1.0)
        window = 0.001
        t_ms   = 500
        t_s    = t_ms / 1000.0
        result = tm.get_packets_at_time(t_ms, time_window=window)
        # Verify against packet log directly
        expected = [s for (t, s) in tm.packet_log if t_s <= t < t_s + window]
        self.assertEqual(result, expected)

    # -- reset ---------------------------------------------------------------
    def test_reset_clears_log(self):
        tm = TrafficModel('CBR', packet_rate=10)
        tm.generate_packets(1.0)
        self.assertGreater(len(tm.packet_log), 0)
        tm.reset()
        self.assertEqual(len(tm.packet_log), 0)
        self.assertEqual(tm.time, 0)


# ===========================================================================
#  Main
# ===========================================================================
if __name__ == '__main__':
    unittest.main(verbosity=2)
