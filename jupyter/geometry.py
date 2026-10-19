"""Sun direction and circular J2 orbit geometry.

sun_direction(time, frame="ecef") -> Earth-to-Sun unit vector.
orbit_position(time, ..., frame="ecef", j2=True) -> geocentric position in km.

Frames: ECI = star-fixed GCRS; ECEF = Earth-fixed ITRS (default).
Scalar time returns (3,); arrays of times return (..., 3).
"""

from pathlib import Path
import sys

import numpy as np
from astropy.coordinates import get_sun, GCRS, ITRS, CartesianRepresentation
from astropy.time import Time
from astropy.utils import iers
from astropy import units as u

# Support running directly from this checkout, like simulate.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Same Earth constants as darknessalp.constants (km and seconds).
from darknessalp.constants import MU_EARTH_KM3_S2, R_EQUATOR_KM, J2


def _time(time):
    return time if isinstance(time, Time) else Time(time, scale="utc")


def _frame(frame):
    frame = frame.lower()
    if frame not in ("eci", "ecef"):
        raise ValueError("frame must be 'eci' (GCRS) or 'ecef' (ITRS)")
    return frame


def sun_direction(time, frame="ecef"):
    """Earth-to-Sun unit vector at UTC time(s), using Astropy's ephemeris.

    Pass an ISO UTC string, a list of strings, or an astropy Time.
    Returns (3,) for one time, or (..., 3) for an array of times.
    ECEF is ITRS; ECI is GCRS. The observer is Earth's center.
    """
    time = _time(time)
    frame = _frame(frame)
    sun = get_sun(time)
    if frame == "ecef":
        # Use bundled orientation data without downloading at runtime.
        with iers.conf.set_temp("auto_download", False), iers.conf.set_temp("auto_max_age", None):
            sun = sun.transform_to(ITRS(obstime=time))
    xyz = np.moveaxis(sun.cartesian.xyz.to_value(u.km), 0, -1)
    return xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)


def orbit_position(time, epoch="2026-10-19T00:00:00", altitude_km=500.0,
                   inclination_deg=97.4, raan_deg=0.0, phase_deg=0.0,
                   frame="ecef", j2=True):
    """Circular orbit with first-order secular J2 drift; orbit position in km from Earth's center.

    time/epoch are UTC ISO strings or astropy Time objects. Altitude is
    above the equatorial Earth radius. RAAN and inclination orient the
    orbit in GCRS; phase is argument of latitude at epoch (degrees).
    Returns (3,) or (..., 3), matching the time input shape.
    J2 drifts RAAN and argument of latitude (mean anomaly + periapsis).
    Set j2=False for two-body motion. No drag or short-period terms.
    """
    frame = _frame(frame)
    params = np.asarray([altitude_km, inclination_deg, raan_deg, phase_deg], dtype=float)
    if params.shape != (4,) or not np.all(np.isfinite(params)):
        raise ValueError("Orbit parameters must be finite scalars")
    if altitude_km < 0 or not 0 <= inclination_deg <= 180:
        raise ValueError("Require altitude_km >= 0 and inclination in [0, 180]")
    time, epoch = _time(time), _time(epoch)
    if not epoch.isscalar:
        raise ValueError("epoch must be a single time")
    radius = R_EQUATOR_KM + altitude_km
    rate = np.sqrt(MU_EARTH_KM3_S2 / radius**3)
    dt = (time - epoch).to_value(u.s)
    inc = np.deg2rad(inclination_deg)
    correction = J2 * rate * (R_EQUATOR_KM / radius)**2 if j2 else 0.0
    node_rate = -1.5 * correction * np.cos(inc)
    # For e=0: u_dot = M_dot + omega_dot.
    latitude_rate = rate + 1.5 * correction * (4*np.cos(inc)**2 - 1)
    phase = np.deg2rad(phase_deg) + latitude_rate * dt
    node = np.deg2rad(raan_deg) + node_rate * dt
    node_axis = np.stack([np.cos(node), np.sin(node), np.zeros_like(node)], axis=-1)
    transverse = np.stack([-np.sin(node)*np.cos(inc),
                           np.cos(node)*np.cos(inc), np.full_like(node, np.sin(inc))], axis=-1)
    xyz = radius * (np.cos(phase)[..., None]*node_axis
                    + np.sin(phase)[..., None]*transverse)
    if frame == "ecef":
        position = GCRS(CartesianRepresentation(np.moveaxis(xyz, -1, 0)*u.km), obstime=time)
        with iers.conf.set_temp("auto_download", False), iers.conf.set_temp("auto_max_age", None):
            position = position.transform_to(ITRS(obstime=time))
        xyz = np.moveaxis(position.cartesian.xyz.to_value(u.km), 0, -1)
    return xyz




if __name__ == "__main__":
    time = "2026-10-19T00:00:00"
    sun_ecef_unit = sun_direction(time)
    position_ecef_km = orbit_position(time)
    print("Sun direction [Earth-fixed ECEF unit]:", sun_ecef_unit)
    print("Orbit position [Earth-fixed ECEF km]:", position_ecef_km)
