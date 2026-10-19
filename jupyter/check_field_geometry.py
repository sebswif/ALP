"""Run requested geometry and IGRF sanity checks from the workspace root."""
import numpy as np
from astropy.coordinates import EarthLocation
from astropy import units as u
from field import line_of_sight, igrf_field
from darknessalp.constants import R_EARTH_KM

TIME = "2026-10-19T00:00:00"
print(f"IGRF-14 epoch: {TIME} UTC")
print("Ray altitude convention: spherical radius 6371.2 km; cutoff 60 km")
radius = R_EARTH_KM + 600.0
shell = R_EARTH_KM + 60.0
position_ecef_km = np.array([radius, 0.0, 0.0])
ratio = shell / radius
looks = {"nadir": np.array([-1.0, 0.0, 0.0]),
         "grazing": np.array([-np.sqrt(1-ratio**2), ratio, 0.0])}
for name, look_ecef_unit in looks.items():
    print(f"\n{name.upper()}")
    result = line_of_sight(position_ecef_km, look_ecef_unit, TIME, samples=501)
    expected = 540.0 if name == "nadir" else np.sqrt(radius**2-shell**2)
    actual = result["distance_km"][-1]
    np.testing.assert_allclose(actual, expected, atol=0.001, rtol=0)
    np.testing.assert_allclose(result["minimum_altitude_km"], 60, atol=1e-7)
    assert result["stopped_at_cutoff"]
    quoted = 540.0 if name == "nadir" else 2640.0
    print(f"Requested reference: {quoted:.1f} km; difference: {actual-quoted:+.3f} km")

print("\nCITY FIELDS: WGS84 geodetic lat/lon, height=0 m (sea level)")
for name, lat, lon, target in [("Boulder",40.015,-105.2705,52.0),
                                ("Sao Paulo",-23.5505,-46.6333,23.0),
                                ("Yakutsk",62.0355,129.6755,60.0)]:
    location = EarthLocation.from_geodetic(lon*u.deg, lat*u.deg, 0*u.m)
    city_ecef_km = np.array([component.to_value(u.km) for component in location.geocentric])
    b_ecef_T = igrf_field(city_ecef_km, TIME, report=True)
    magnitude = np.linalg.norm(b_ecef_T)*1e6
    print(f"{name}: lat={lat}, lon={lon}; |B|={magnitude:.3f} uT; "
          f"reference={target:.1f} uT; difference={magnitude-target:+.3f} uT "
          f"({100*(magnitude/target-1):+.2f}%)")
print("\nAnalytic path checks passed. City references are comparisons, not forced matches.")
