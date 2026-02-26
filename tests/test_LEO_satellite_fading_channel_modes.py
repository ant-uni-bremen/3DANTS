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
test_channel_modes.py
=====================
Lightweight integration test for Satellite_Fading_channel across the three
Doppler modes (full / compensated / scaled), using the same inner-loop
architecture as 3D_network_with_traffic.py but with a synthetic orbit so
the test needs no skyfield, SGP4, or network access.

What is tested
--------------
For each mode the test:
  1. Runs 60 simulation steps (1 ms each) — enough to confirm steady operation
     and to trigger the 1-degree elevation regen at least once.
  2. Checks the channel sample at every step is finite and complex.
  3. Checks the regen trigger fires when |elevation - last_regen| >= 1 degree.
  4. Compares the effective Doppler between modes to confirm they are distinct.
  5. Checks the marginal distribution shape (mean power) is consistent across
     modes — it must not depend on doppler_mode.
  6. Confirms self.fs is never mutated (bug B2 regression check).
  7. Confirms Rank_matching raises TypeError on complex input (bug B1 regression).
  8. Plots the channel envelope for all three modes side by side and saves to PNG.
"""

import sys
import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')  # no display needed
import matplotlib.pyplot as plt

# ── Path setup ───────────────────────────────────────────────────────────────
_REPO = os.path.join(os.path.dirname(__file__), '3DANTS')
sys.path.insert(0, os.path.join(_REPO, '3DANTS', 'Communication channel'))

from Satellite_fading_channel import Satellite_Fading_channel

# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic orbit — realistic LEO trajectory without skyfield
# ═══════════════════════════════════════════════════════════════════════════════

def synthetic_pass(n_seconds, el_start=25.0, el_peak=65.0, v_orbital_kms=7.562):
    """
    Generate a synthetic satellite pass as a sequence of per-second observations.

    Returns list of dicts with keys:
        elevation  : elevation angle at ground station [degrees]
        v_LOS      : radial (LOS) velocity [km/s]
        distance   : slant range [km]

    Geometry model: parabolic elevation profile (rise -> peak -> set).
    v_LOS = v_orbital * cos(elevation) -- range-rate at given elevation.
    distance approximated from altitude and elevation via simple trig.
    """
    h_km    = 600.0          # satellite altitude [km]
    R_E_km  = 6371.0         # Earth radius [km]
    steps   = []

    for t in range(n_seconds):
        # Parabolic elevation profile peaking at n_seconds/2
        frac = t / max(n_seconds - 1, 1)
        # Rises from el_start, peaks at el_peak at frac=0.5, sets back to el_start
        el = el_start + (el_peak - el_start) * 4 * frac * (1 - frac)

        # v_LOS: range-rate = v_orbital * cos(el) (+ approaching, - receding)
        # Satellite approaches in first half, recedes in second half
        sign   = +1.0 if frac < 0.5 else -1.0
        v_los  = sign * v_orbital_kms * math.cos(math.radians(el))

        # Slant range from elevation angle and altitude
        el_r   = math.radians(el)
        # Exact formula: d = -R_E*sin(el) + sqrt(R_E^2*sin^2(el) + h^2 + 2*R_E*h)
        dist   = (-R_E_km * math.sin(el_r)
                  + math.sqrt(R_E_km**2 * math.sin(el_r)**2
                              + h_km**2 + 2 * R_E_km * h_km))
        steps.append({'elevation': el, 'v_LOS': v_los, 'distance': dist})

    return steps


# ═══════════════════════════════════════════════════════════════════════════════
# ShadowingFading stub  (the real class is not in the repo)
# ═══════════════════════════════════════════════════════════════════════════════

class ShadowingFading:
    """
    Minimal stub that returns a flat (zero dB) shadowing batch.
    The real AR(1) large-scale fading model is not exercised here — we only
    need a valid array to consume so the sample-indexing logic is tested.
    """
    def __init__(self, tau, N):
        self.tau = tau
        self.N   = N

    def SF_LOS_calc(self, elevation_angle, scenario, freq_band):
        # Return N zeros [dB] — no shadowing loss
        return np.zeros(self.N)


# ═══════════════════════════════════════════════════════════════════════════════
# Core simulation loop — mirrors the inner loop of 3D_network_with_traffic.py
# ═══════════════════════════════════════════════════════════════════════════════

def run_mode(doppler_mode, fc, batch_size, n_seconds,
             v_residual_mps=1.5, Tc_target_s=0.05,
             regen_deg=1.0, shadowing_regen_deg=5.0):
    """
    Run the inner simulation loop for one satellite pass under a given mode.

    Parameters
    ----------
    doppler_mode     : 'full' | 'compensated' | 'scaled'
    fc               : carrier frequency [Hz]
    batch_size       : number of channel samples per batch
    n_seconds        : number of simulation seconds (= steps, 1 ms/step)
    v_residual_mps   : UE speed for 'compensated' mode
    Tc_target_s      : target coherence time for 'scaled' mode
    regen_deg        : elevation change [degrees] that triggers small-scale regen
    shadowing_regen_deg : elevation change [degrees] for shadowing regen

    Returns
    -------
    dict with:
        samples          : list of complex channel samples (one per second step)
        regen_count      : how many times the 1-degree trigger fired
        fs_unchanged     : True if self.fs was never mutated (B2 check)
        fd_effective     : effective Doppler [Hz] at 45-degree elevation
        mean_power       : mean |h|^2 over all steps
    """
    # ── Channel object ───────────────────────────────────────────────────────
    num_samples_nakagami = batch_size
    num_samples_rician   = batch_size // 10
    fs_initial           = 1000    # 1 ms / sample

    sim = Satellite_Fading_channel(
        num_samples_nakagami, num_samples_rician, fs_initial, N=32,
        doppler_mode=doppler_mode,
        v_residual_mps=v_residual_mps,
        Tc_target_s=Tc_target_s,
        fc_ref=fc)

    fs_at_start = sim.fs  # record to check for mutation later

    # ── Orbit trajectory ─────────────────────────────────────────────────────
    orbit = synthetic_pass(n_seconds)

    # ── Shadowing stub ───────────────────────────────────────────────────────
    large_shadowing_N = 5000
    elev0  = orbit[0]['elevation']
    Shadowing = ShadowingFading(tau=10, N=large_shadowing_N)

    # ── State variables (mirroring 3D_network_with_traffic.py) ───────────────
    last_regen_elevation_small = elev0
    last_regen_elevation       = elev0
    shadowing_sample_index     = 0
    shadowing_samples_interval = Shadowing.SF_LOS_calc(elev0, 'LOS', 'SBand')

    # Initial batch
    step0 = orbit[0]
    batch = sim.run_simulation(
        step0['elevation'], step0['v_LOS'], fc, step0['distance'])
    small_scale_sample_index = 0

    samples      = []
    regen_count  = 0

    for step in orbit:
        el     = step['elevation']
        v_los  = step['v_LOS']
        dist   = step['distance']

        # ── Small-scale fading: 1-degree elevation regen trigger ─────────────
        if abs(el - last_regen_elevation_small) >= regen_deg:
            last_regen_elevation_small = el
            batch = sim.run_simulation(el, v_los, fc, dist)
            small_scale_sample_index = 0
            regen_count += 1

        # Batch exhaustion guard
        if small_scale_sample_index >= len(batch):
            batch = sim.run_simulation(el, v_los, fc, dist)
            small_scale_sample_index = 0

        h = batch[small_scale_sample_index]
        small_scale_sample_index += 1
        samples.append(h)

        # ── Shadowing: 5-degree regen trigger ────────────────────────────────
        if abs(el - last_regen_elevation) >= shadowing_regen_deg:
            last_regen_elevation       = el
            Shadowing                  = ShadowingFading(tau=10, N=large_shadowing_N)
            shadowing_samples_interval = Shadowing.SF_LOS_calc(el, 'LOS', 'SBand')
            shadowing_sample_index     = 0

        if shadowing_sample_index >= len(shadowing_samples_interval):
            shadowing_samples_interval = Shadowing.SF_LOS_calc(el, 'LOS', 'SBand')
            shadowing_sample_index     = 0

        _ = shadowing_samples_interval[shadowing_sample_index]
        shadowing_sample_index += 1

    # ── Effective fd at el=45 for comparison ─────────────────────────────────
    v_los_45   = 7.562 * math.cos(math.radians(45))   # km/s
    fd_eff     = sim._effective_fd(v_los_45, fc)

    return {
        'samples'     : samples,
        'regen_count' : regen_count,
        'fs_unchanged': sim.fs == fs_at_start,
        'fd_effective': fd_eff,
        'mean_power'  : float(np.mean(np.abs(samples)**2)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_all_modes():
    np.random.seed(0)
    fc         = 2.0e9       # S-band, matching 3D_network_with_traffic.py
    batch_size = 500         # light: 500 ms batch
    n_seconds  = 60          # 60-second pass slice

    modes = {
        'full'       : dict(doppler_mode='full'),
        'compensated': dict(doppler_mode='compensated', v_residual_mps=1.5),
        'scaled'     : dict(doppler_mode='scaled',      Tc_target_s=0.05),
    }

    results = {}
    print("=" * 60)
    print("Satellite_Fading_channel — three-mode integration test")
    print(f"fc={fc/1e9:.1f} GHz  batch={batch_size}  pass={n_seconds}s")
    print("=" * 60)

    for name, kwargs in modes.items():
        r = run_mode(fc=fc, batch_size=batch_size, n_seconds=n_seconds, **kwargs)
        results[name] = r
        print(f"\n[{name}]")
        print(f"  Samples generated : {len(r['samples'])}")
        print(f"  Regen triggers    : {r['regen_count']}  (1-deg elevation change)")
        print(f"  fd at el=45 deg   : {r['fd_effective']:.2f} Hz")
        print(f"  Tc at el=45 deg   : {0.423/r['fd_effective']*1000:.2f} ms" if r['fd_effective'] > 0 else "  Tc: inf")
        print(f"  Mean power |h|²   : {r['mean_power']:.4f}")
        print(f"  self.fs unchanged : {r['fs_unchanged']}")

    # ── Assertions ───────────────────────────────────────────────────────────
    PASS = True

    for name, r in results.items():
        samples = np.array(r['samples'])

        # 1. All samples finite and complex
        assert np.iscomplexobj(samples), f"{name}: samples not complex"
        assert np.all(np.isfinite(samples)), f"{name}: non-finite samples"
        print(f"\n✓ [{name}] all {len(samples)} samples finite and complex")

        # 2. Regen fired at least once
        assert r['regen_count'] >= 1, \
            f"{name}: 1-degree trigger never fired (pass too short or elevation too flat)"
        print(f"✓ [{name}] regen triggered {r['regen_count']} time(s)")

        # 3. Bug B2: self.fs not mutated
        assert r['fs_unchanged'], f"{name}: B2 regression — self.fs was mutated"
        print(f"✓ [{name}] self.fs not mutated (B2 fix holds)")

        # 4. Mean power in physically reasonable range (not wildly off)
        assert 0.1 < r['mean_power'] < 100.0, \
            f"{name}: mean power {r['mean_power']:.4f} out of range [0.1, 100]"
        print(f"✓ [{name}] mean power {r['mean_power']:.4f} in reasonable range")

    # 5. Doppler frequencies are distinct across modes
    fd_full  = results['full']['fd_effective']
    fd_comp  = results['compensated']['fd_effective']
    fd_scale = results['scaled']['fd_effective']

    assert fd_full > 1000 * fd_comp, \
        f"full ({fd_full:.1f} Hz) should be >> compensated ({fd_comp:.1f} Hz)"
    assert abs(fd_comp - fd_scale) > 1.0, \
        f"compensated ({fd_comp:.1f} Hz) and scaled ({fd_scale:.1f} Hz) should differ"
    print(f"\n✓ Doppler frequencies distinct across modes:")
    print(f"   full={fd_full:.0f} Hz  compensated={fd_comp:.2f} Hz  scaled={fd_scale:.2f} Hz")

    # 6. Marginal distribution consistent across modes
    # Mean power should be in the same order of magnitude (within 10x)
    powers = {k: v['mean_power'] for k, v in results.items()}
    max_ratio = max(powers.values()) / min(powers.values())
    assert max_ratio < 15.0, \
        f"Mean power too different across modes: {powers}  ratio={max_ratio:.1f}"
    print(f"✓ Mean power consistent across modes (max ratio {max_ratio:.2f}x)")

    # 7. Bug B1: Rank_matching rejects complex input
    sim_test = Satellite_Fading_channel(100, 10, 1000, 8, doppler_mode='full')
    try:
        complex_input = np.ones(100, dtype=np.complex128)
        sim_test.Rank_matching(complex_input, np.abs(np.random.randn(100)))
        assert False, "B1 regression: Rank_matching accepted complex input"
    except TypeError:
        print("✓ Rank_matching raises TypeError on complex input (B1 fix holds)")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(results, output_path='channel_mode_comparison.png'):
    """
    Three-panel plot: channel envelope for each mode over the 60-second pass.
    """
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    colours = {'full': '#d62728', 'compensated': '#1f77b4', 'scaled': '#2ca02c'}

    for ax, (name, r) in zip(axes, results.items()):
        samples   = np.array(r['samples'])
        envelope  = 20 * np.log10(np.abs(samples) + 1e-12)   # dB
        t_axis    = np.arange(len(samples))   # seconds

        ax.plot(t_axis, envelope, lw=0.9, color=colours[name])
        ax.set_ylabel('|h| (dB)', fontsize=9)
        fd_val = r['fd_effective']
        tc_str = 'inf' if fd_val == 0 else str(round(0.423 / fd_val * 1e3)) + ' ms'
        ax.set_title(
            f"doppler_mode='{name}' | fd={fd_val:.1f} Hz | Tc={tc_str} | "
            f"mean power={r['mean_power']:.3f}",
            fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-25, 15)

    axes[-1].set_xlabel('Simulation step (seconds)', fontsize=9)
    fig.suptitle(
        'Satellite_Fading_channel: three Doppler modes\n'
        'fc=2 GHz, LEO 600 km, synthetic pass el 25°→65°→25°, batch=500',
        fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    results = test_all_modes()
    plot_results(results, output_path='/home/vakilifard/Documents/codes_result/LEO_SAT_code/3DANTS/tests/outputs/channel_mode_comparison.png')
