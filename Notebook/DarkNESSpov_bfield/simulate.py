"""Visualize the project's IGRF-14 coefficients in Earth-fixed coordinates."""
from pathlib import Path
import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d.art3d import Line3DCollection

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from darknessalp.constants import R_EARTH_KM
from darknessalp.field.igrf import load_igrf, igrf_field


def trace_field_lines(seeds, coeffs, max_radius=4.0, step=0.015, max_steps=2000):
    """Trace dr/ds = +/- B/|B| using batched RK4 in Earth-radius units.

    Return one B-oriented polyline per seed plus its two endpoint statuses.
    Stop at the reference surface, outer sphere, or the integration budget.
    """
    seeds = np.asarray(seeds, dtype=float)
    if seeds.ndim != 2 or seeds.shape[1] != 3 or not np.isfinite(seeds).all():
        raise ValueError("seeds must be a finite (N, 3) array")
    radii = np.linalg.norm(seeds, axis=1)
    if not (np.isfinite(max_radius) and max_radius > 1 and
            np.isfinite(step) and 0 < step <= 0.1 and max_steps > 0):
        raise ValueError("invalid tracing limits")
    if np.any((radii <= 1) | (radii >= max_radius)):
        raise ValueError("seeds must lie strictly between the two boundary spheres")

    def direction(points):
        b = igrf_field(points * R_EARTH_KM, coeffs)
        norm = np.linalg.norm(b, axis=1, keepdims=True)
        if not np.isfinite(b).all() or np.any(norm == 0):
            raise ValueError("nonfinite or zero field encountered")
        return b / norm

    branches, statuses = [], []
    for sign in (-1, 1):
        points = seeds.copy()
        paths = [[point.copy()] for point in points]
        active = np.ones(len(points), dtype=bool)
        status = np.full(len(points), "step_limit", dtype="U12")
        ds = sign * step
        for _ in range(max_steps):
            ids = np.flatnonzero(active)
            if not len(ids):
                break
            old = points[ids]
            k1 = direction(old)
            k2 = direction(old + ds*k1/2)
            k3 = direction(old + ds*k2/2)
            k4 = direction(old + ds*k3)
            new = old + ds*(k1 + 2*k2 + 2*k3 + k4)/6
            radius = np.linalg.norm(new, axis=1)
            for j, index in enumerate(ids):
                boundary = 1.0 if radius[j] <= 1 else max_radius if radius[j] >= max_radius else None
                if boundary is not None:
                    # Intersect the final segment with the stopping sphere.
                    delta = new[j] - old[j]
                    roots = np.roots([delta @ delta, 2*old[j] @ delta,
                                      old[j] @ old[j] - boundary**2])
                    fraction = min(float(v.real) for v in roots
                                   if abs(v.imag) < 1e-9 and -1e-9 <= v.real <= 1+1e-9)
                    new[j] = old[j] + np.clip(fraction, 0, 1)*delta
                    active[index] = False
                    status[index] = "surface" if boundary == 1 else "outer_limit"
                paths[index].append(new[j].copy())
            points[ids] = new
        branches.append(paths)
        statuses.append(status)
    lines = [np.concatenate((np.asarray(a)[::-1], np.asarray(b)[1:]))
             for a, b in zip(*branches)]
    return lines, np.column_stack(statuses)


def field_line_figure(coeffs, year, output_dir, max_radius=4.0,
                      line_count=720, shell_ax=None):
    """Draw a dense, approximately equal-area sample of full-vector field lines."""
    if isinstance(line_count, bool) or not isinstance(line_count, (int, np.integer)) or line_count < 1:
        raise ValueError("line_count must be a positive integer")
    # Fibonacci sphere: cover both hemispheres without clustering at the poles.
    index = np.arange(line_count)
    z = 1 - 2*(index + 0.5)/line_count
    longitude = index*np.pi*(3-np.sqrt(5))
    rho = np.sqrt(1-z*z)
    seeds = 1.02*np.column_stack((rho*np.cos(longitude), rho*np.sin(longitude), z))
    print(f"Tracing {line_count} field lines over the full globe...", flush=True)
    lines, statuses = trace_field_lines(seeds, coeffs, max_radius=max_radius)
    segments = np.concatenate([np.stack((line[:-1], line[1:]), axis=1) for line in lines])
    # Bound Legendre-array memory when coloring a dense collection.
    midpoints = segments.mean(axis=1)*R_EARTH_KM
    values = np.concatenate([np.linalg.norm(igrf_field(batch, coeffs), axis=1)*1e6
                             for batch in np.array_split(midpoints, max(1, int(np.ceil(len(midpoints)/4096))))])
    fig = plt.figure(figsize=(11, 10))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    collection = Line3DCollection(segments, cmap="plasma", norm=LogNorm(values.min(), values.max()),
                                  linewidth=0.45, alpha=0.55)
    collection.set_array(values)
    ax.add_collection3d(collection)
    u, v = np.meshgrid(np.linspace(0, 2*np.pi, 80), np.linspace(0, np.pi, 40))
    ax.plot_surface(np.sin(v)*np.cos(u), np.sin(v)*np.sin(u), np.cos(v),
                    color="steelblue", alpha=0.4, linewidth=0, shade=True, zorder=0)
    limit = max_radius*1.03
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit),
           xlabel="ECEF x / Earth radius", ylabel="ECEF y / Earth radius",
           zlabel="ECEF z / Earth radius",
           title=f"IGRF-14 | {line_count} field lines | {year:.2f}\n"
                 f"Internal field only; traces stop at the surface or {max_radius:g} Earth radii (or step limit)")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=20, azim=35)
    if shell_ax is not None:
        # Reuse the same full-vector trajectories on the colored altitude shell.
        shell_ax.add_collection3d(Line3DCollection(
            segments, colors="dimgray", linewidth=0.4, alpha=0.4, zorder=2))
        shell_ax.view_init(elev=20, azim=35)
        shell_limit = max(limit, np.max(np.abs(shell_ax.get_xlim())))
        shell_ax.set(xlim=(-shell_limit, shell_limit), ylim=(-shell_limit, shell_limit),
                     zlim=(-shell_limit, shell_limit))
    fig.colorbar(collection, ax=ax, shrink=0.65, label="Field magnitude (microtesla, log scale)")
    np.savez_compressed(output_dir / "field_lines.npz", year=year, max_radius=max_radius,
                        seeds_earth_radii=seeds, endpoint_status=statuses,
                        **{f"line_{i:03d}_ecef_km": line*R_EARTH_KM for i, line in enumerate(lines)})
    unique, counts = np.unique(statuses, return_counts=True)
    print(f"Traced {len(lines)} 3D field lines; endpoints: {dict(zip(unique, counts.tolist()))}")
    return fig


def simulate(year=2026.72, altitude_km=500.0, output_dir=None,
             line_count=720):
    """Return figures and save maps plus ECEF samples; altitude is spherical."""
    if not np.isfinite(altitude_km) or altitude_km < 0:
        raise ValueError("altitude_km must be finite and nonnegative")
    coeffs = load_igrf(year)
    output_dir = Path(output_dir or ROOT / "outputs" / "DarkNESSpov_bfield")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Cell centers avoid the existing evaluator's exact geographic-pole singularity.
    lon, lat = np.meshgrid(np.linspace(-179, 179, 180), np.linspace(-89, 89, 90))
    lam, phi = np.deg2rad(lon), np.deg2rad(lat)
    unit = np.stack((np.cos(phi)*np.cos(lam), np.cos(phi)*np.sin(lam), np.sin(phi)), axis=-1)
    xyz = unit.reshape(-1, 3) * (R_EARTH_KM + altitude_km)
    field = igrf_field(xyz, coeffs) * 1e6
    strength = np.linalg.norm(field, axis=1).reshape(lat.shape)
    fig_map, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    im = ax.pcolormesh(lon, lat, strength, shading="auto", cmap="viridis")
    fig_map.colorbar(im, ax=ax, label="Field magnitude (µT)")
    ax.set(xlabel="East longitude (degrees)", ylabel="Geocentric latitude (degrees)",
           title=f"IGRF-14 | {year:.2f} | {altitude_km:g} km above reference sphere",
           xlim=(-180, 180), ylim=(-90, 90))

    fig_3d = plt.figure(figsize=(11, 10))
    ax3 = fig_3d.add_subplot(111, projection="3d", computed_zorder=False)
    norm = Normalize(strength.min(), strength.max())
    # Colored shell is at the selected altitude; coordinates are Earth radii.
    shell = xyz.reshape(*lat.shape, 3) / R_EARTH_KM
    ax3.plot_surface(*np.moveaxis(shell, -1, 0), facecolors=plt.cm.viridis(norm(strength)),
                     rstride=2, cstride=2, shade=False, alpha=0.65, zorder=0)
    fig_3d.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax3,
                   shrink=0.6, label="Field magnitude (µT)")
    ax3.set(xlabel="ECEF x / Rₑ", ylabel="ECEF y / Rₑ", zlabel="ECEF z / Rₑ",
            title=f"IGRF-14 field lines and colored altitude shell | {year:.2f}\n"
                  f"Shell: {altitude_km:g} km; gray curves: 3D field lines truncated at 4 Earth radii")
    ax3.set_box_aspect((1, 1, 1))

    # An even grid avoids x=y=0, where the evaluator uses a pole fallback.
    grid = np.linspace(-3, 3, 240)
    xx, zz = np.meshgrid(grid, grid)
    points = np.column_stack((xx.ravel(), np.zeros(xx.size), zz.ravel()))
    outside = np.linalg.norm(points, axis=1) >= 1.0
    b = np.full_like(points, np.nan)
    b[outside] = igrf_field(points[outside] * R_EARTH_KM, coeffs) * 1e6
    bx, by, bz = b.reshape(*xx.shape, 3).transpose(2, 0, 1)
    magnitude = np.sqrt(bx**2 + by**2 + bz**2)
    fig_slice, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    stream = ax.streamplot(grid, grid, np.ma.masked_invalid(bx), np.ma.masked_invalid(bz),
                           color=np.ma.masked_invalid(np.log10(magnitude)),
                           cmap="plasma", density=1.6, linewidth=0.8)
    fig_slice.colorbar(stream.lines, ax=ax, label="log₁₀(|B| / µT)")
    ax.add_patch(Circle((0, 0), 1, facecolor="steelblue", edgecolor="black"))
    ax.set(xlabel="ECEF x / Rₑ", ylabel="ECEF z / Rₑ", aspect="equal",
           title="IGRF-14: projected field in the y = 0 plane\nStreamlines of (Bx, Bz), not full 3D field lines")

    np.savez_compressed(output_dir / "field_samples.npz", year=year, altitude_km=altitude_km,
                        latitude_deg=lat, longitude_deg=lon, position_ecef_km=xyz,
                        field_ecef_uT=field, magnitude_uT=strength)
    figures = {"global_map": fig_map, "earth_3d": fig_3d, "field_cross_section": fig_slice,
               "field_lines_3d": field_line_figure(coeffs, year, output_dir,
                   line_count=line_count, shell_ax=ax3)}
    for name, figure in figures.items():
        figure.savefig(output_dir / f"{name}.png", dpi=180)
    print(f"IGRF-14 {year:.2f}, altitude {altitude_km:g} km: "
          f"{strength.min():.2f}–{strength.max():.2f} µT")
    print(f"Saved plots and field_samples.npz to {output_dir}")
    return figures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=float, default=2026.72)
    parser.add_argument("--altitude-km", type=float, default=500.0)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--line-count", type=int, default=720)
    args = parser.parse_args()
    simulate(args.year, args.altitude_km, line_count=args.line_count)
    if args.show:
        plt.show()
    else:
        plt.close("all")
