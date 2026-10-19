"""Quantitative magnetic field samples: IGRF-14, Earth-fixed ECEF only."""
from pathlib import Path
import sys
import numpy as np
from astropy.time import Time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from darknessalp.field.igrf import load_igrf, igrf_field as _igrf_field
from darknessalp.field.dipole import dipole_field as _dipole_field
from darknessalp.constants import R_EARTH_KM


def igrf_field(position_ecef_km, time, report=True):
    """Return IGRF-14 B [Earth-fixed ECEF tesla], shape matching (..., 3).

    time is a scalar UTC ISO date or astropy Time. Positions are km from
    Earth's center; suitable for one snapshot along a geometry ray.
    """
    position = np.asarray(position_ecef_km, dtype=float)
    if position.ndim < 1 or position.shape[-1] != 3 or not np.isfinite(position).all():
        raise ValueError("Require finite (..., 3) ECEF positions in km")
    if np.any(np.linalg.norm(position, axis=-1) == 0):
        raise ValueError("Field undefined at Earth's center")
    epoch = time if isinstance(time, Time) else Time(time, scale="utc")
    if not epoch.isscalar:
        raise ValueError("Use one epoch per field snapshot")
    result = _igrf_field(position.reshape(-1, 3), load_igrf(epoch.decimalyear)).reshape(position.shape)
    if report:
        print(f"Model: IGRF-14; UTC {epoch.utc.isot}")
        print("Positions [Earth-fixed ECEF km]:", position)
        print("B [Earth-fixed ECEF tesla]:", result)
        print("|B| [tesla]:", np.linalg.norm(result, axis=-1))
    return result


# Backward-compatible name for existing notebooks.
field_ecef = igrf_field


def dipole_field(position_ecef_km, time, report=True):
    """Tilted IGRF degree-1 dipole for SHAPES ONLY; ECEF km -> ECEF tesla.

    Use igrf_field for quoted field strengths and conversion calculations.
    """
    position = np.asarray(position_ecef_km, dtype=float)
    if position.ndim < 1 or position.shape[-1] != 3 or not np.isfinite(position).all():
        raise ValueError("Require finite (..., 3) ECEF positions in km")
    if np.any(np.linalg.norm(position, axis=-1) == 0):
        raise ValueError("Field undefined at Earth's center")
    epoch = time if isinstance(time, Time) else Time(time, scale="utc")
    if not epoch.isscalar:
        raise ValueError("Use one epoch per field snapshot")
    result = _dipole_field(position.reshape(-1, 3), load_igrf(epoch.decimalyear)).reshape(position.shape)
    if report:
        print("Model: tilted degree-1 dipole, visualization only")
        print("Positions [Earth-fixed ECEF km]:", position)
        print("B [Earth-fixed ECEF tesla]:", result)
    return result


def line_of_sight(position_ecef_km, look_ecef_unit, time,
                  max_distance_km=10000.0, samples=501, report=True):
    """Sample IGRF along a source-facing ECEF ray, stopping at 60 km.

    Returns geometry, B_ecef_T, B_perp_ecef_T, and the coherent transverse
    VECTOR integral in T m. Integrate the vector before taking its norm.
    This is a frozen field snapshot with zero phase mismatch (q=0).
    The ray points FROM spacecraft toward source; photons propagate -look.
    Zenith: 0=outward, 90=horizontal, 180=nadir. Rays missing the 60 km
    shell stop at max_distance_km. Computed angles print when report=True.
    """
    cutoff_altitude_km = 60.0
    r = np.asarray(position_ecef_km, dtype=float)
    look = np.asarray(look_ecef_unit, dtype=float)
    if r.shape != (3,) or look.shape != (3,) or not np.isfinite([r, look]).all():
        raise ValueError("Require finite (3,) ECEF position and look arrays")
    if np.linalg.norm(look) == 0:
        raise ValueError("Look direction cannot be zero")
    look = look / np.linalg.norm(look)
    if (not np.isfinite([max_distance_km, cutoff_altitude_km]).all()
            or max_distance_km <= 0 or cutoff_altitude_km < 0
            or not isinstance(samples, (int, np.integer)) or samples < 2):
        raise ValueError("Require positive range, nonnegative cutoff, and >=2 samples")
    shell = R_EARTH_KM + cutoff_altitude_km
    radius = np.linalg.norm(r)
    if radius <= shell:
        raise ValueError("Spacecraft must be above the absorption shell")
    along = np.dot(r, look)
    discriminant = along**2 - (radius**2 - shell**2)
    end = max_distance_km
    stopped = False
    if along < 0 and discriminant >= -1e-8:
        entry = -along - np.sqrt(max(0.0, discriminant))
        if entry <= end:
            end, stopped = entry, True
    distance_km = np.linspace(0, end, samples)
    path_ecef_km = r + distance_km[:, None]*look
    zenith = np.degrees(np.arccos(np.clip(along/radius, -1, 1)))
    closest_s = np.clip(-along, 0, end)
    min_altitude = np.linalg.norm(r + closest_s*look) - R_EARTH_KM
    if report:
        print("Spacecraft [Earth-fixed ECEF km]:", r)
        print("Look toward source [Earth-fixed ECEF unit]:", look)
        print("Photon propagation [Earth-fixed ECEF unit]:", -look)
        print(f"Returned zenith={zenith:.6f} deg; nadir angle={180-zenith:.6f} deg")
        print(f"Returned path length={end:.6f} km; minimum altitude={min_altitude:.6f} km")
        print(f"Stopped at {cutoff_altitude_km:g} km shell: {stopped}")
        print("Path endpoint [Earth-fixed ECEF km]:", path_ecef_km[-1])
    ray = dict(distance_km=distance_km, path_ecef_km=path_ecef_km,
                look_ecef_unit=look, zenith_deg=zenith,
                minimum_altitude_km=min_altitude, stopped_at_cutoff=stopped)

    b_ecef_T = igrf_field(ray["path_ecef_km"], time, report=False)
    look = ray["look_ecef_unit"]
    b_perp_ecef_T = b_ecef_T - (b_ecef_T @ look)[:, None]*look
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    integral_ecef_Tm = integrate(b_perp_ecef_T, ray["distance_km"]*1000, axis=0)
    amplitude_Tm = float(np.linalg.norm(integral_ecef_Tm))
    if report:
        print("Field model: IGRF-14; all field arrays Earth-fixed ECEF tesla")
        print("B at path endpoints [Earth-fixed ECEF tesla]:", b_ecef_T[[0, -1]])
        print("Integral B_perp ds [Earth-fixed ECEF T m]:", integral_ecef_Tm)
        print("Coherent transverse amplitude [T m]:", amplitude_Tm)
    return dict(**ray, B_ecef_T=b_ecef_T, B_perp_ecef_T=b_perp_ecef_T,
                integral_ecef_Tm=integral_ecef_Tm, amplitude_Tm=amplitude_Tm)
