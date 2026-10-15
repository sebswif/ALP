"""A frozen circular Sun-synchronous orbit for the magnetic-field snapshot."""
import numpy as np
from astropy.time import Time
from astropy.utils import iers

from darknessalp.constants import J2, MU_EARTH_KM3_S2, R_EQUATOR_KM, R_EARTH_KM
from darknessalp.frames.sun import sun_vector
from darknessalp.frames.eci_ecef import eci_to_ecef


def noon_sso(year, altitude_km=500.0, phase_deg=45.0):
    """Return an epoch-fixed ECEF orbit, using mean solar LTAN = 12 h.

    Altitude is above the equatorial radius. Inclination matches first-order
    J2 nodal precession to one revolution per tropical year. This is orbit
    geometry at one epoch, not a propagated Earth-fixed ground track.
    """
    if not np.isfinite(altitude_km) or not 160 <= altitude_km <= 2000:
        raise ValueError("Spacecraft LEO altitude must be between 160 and 2000 km")
    if not np.isfinite(phase_deg):
        raise ValueError("Spacecraft phase must be finite")
    epoch = Time(year, format="decimalyear", scale="utc")
    a = R_EQUATOR_KM + altitude_km
    n = np.sqrt(MU_EARTH_KM3_S2/a**3)
    sun_rate = 2*np.pi/(365.2422*86400)
    inclination = np.arccos(-sun_rate/(1.5*J2*n*(R_EQUATOR_KM/a)**2))
    # Mean local solar time = UT1 hours + east longitude / 15 degrees.
    # Use bundled Earth orientation data, without a network fetch.
    with iers.conf.set_temp("auto_download", False), iers.conf.set_temp("auto_max_age", None):
        ut_hours = ((epoch.ut1.jd + 0.5) % 1)*24
        sun = eci_to_ecef(sun_vector(epoch), epoch)[0]
    node_lon = np.deg2rad((12-ut_hours)*15)
    node = np.array([np.cos(node_lon), np.sin(node_lon), 0.0])
    transverse = np.array([-np.sin(node_lon)*np.cos(inclination),
                            np.cos(node_lon)*np.cos(inclination), np.sin(inclination)])
    def position(phase):
        phase = np.atleast_1d(np.deg2rad(phase))
        return a*(np.cos(phase)[:, None]*node + np.sin(phase)[:, None]*transverse)
    return dict(orbit_ecef_km=position(np.linspace(0, 360, 361)),
                spacecraft_ecef_km=position(phase_deg)[0], ascending_node_ecef_km=a*node,
                sun_ecef_unit=sun/np.linalg.norm(sun), epoch_utc=epoch.isot,
                altitude_km=altitude_km, inclination_deg=np.rad2deg(inclination),
                period_minutes=2*np.pi/n/60, ltan_hours=12.0,
                node_longitude_deg=np.rad2deg(node_lon), ut1_hours=ut_hours,
                phase_deg=phase_deg)


def draw_spacecraft(ax, orbit):
    """Overlay the frozen orbit and a deliberately enlarged spacecraft marker."""
    path = orbit['orbit_ecef_km']/R_EARTH_KM
    craft = orbit['spacecraft_ecef_km']/R_EARTH_KM
    node = orbit['ascending_node_ecef_km']/R_EARTH_KM
    ax.plot(*path.T, color='limegreen', linewidth=2.4, zorder=4, label=f"{orbit['altitude_km']:g} km SSO")
    ax.scatter(*craft, marker='*', s=200, color='gold', edgecolor='black',
               depthshade=False, zorder=5, label='Spacecraft (marker enlarged)')
    ax.scatter(*node, marker='o', s=35, color='limegreen', edgecolor='black',
               depthshade=False, label='Ascending node: LTAN 12:00')
    ax.text(*(craft*1.13), 'Spacecraft', fontsize=9)
    sun = orbit['sun_ecef_unit']
    ax.quiver(*(sun*1.15), *sun, length=0.65, color='darkorange', linewidth=2)
    ax.text(*(sun*1.9), 'Sun', color='darkorange')
    ax.legend(loc='upper left', fontsize=8)
    # Look toward the spacecraft hemisphere so the marker is visible.
    ax.view_init(elev=np.rad2deg(np.arctan2(craft[2], np.hypot(*craft[:2]))),
                 azim=np.rad2deg(np.arctan2(craft[1], craft[0]))+15)
