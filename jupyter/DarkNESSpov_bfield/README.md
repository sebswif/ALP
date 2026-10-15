# Dense IGRF-14 field lines

Open `earth_magnetic_field.ipynb` and run all cells. The spacecraft and orbit
are removed from the active visualization. Existing notebook outputs were
cleared so that earlier spacecraft pictures are not displayed as current results.

The default is **720 field lines**, seeded approximately uniformly over the
whole globe using a Fibonacci sphere, including near-polar regions. There
are infinitely many field lines; any plot shows a finite sample. Sampling
density here is for visualization and does not represent magnetic flux.
Change `LINE_COUNT` in the notebook, or run from the workspace root:

```powershell
python ALP/Notebook/DarkNESSpov_bfield/simulate.py --line-count 720 --show
```

`YEAR` selects decimal year (1900-2030); `ALTITUDE_KM` controls the colored
shell and global map, not field-line seeds. More lines increase runtime and
visual overlap. Dependencies: numpy, matplotlib, and astropy (used by the
existing project field module). No coefficient download is needed.

Generated files in `ALP/outputs/DarkNESSpov_bfield/`:

- `earth_3d.png`: gray field lines surrounding the colored altitude shell.
- `field_lines_3d.png`: field lines colored by local magnitude on a log scale.
- `global_map.png`: magnitude on the selected spherical altitude shell.
- `field_cross_section.png`: projected streamlines in the ECEF y=0 plane.
- `field_lines.npz`: ECEF trajectories in km, seeds in Earth radii, endpoint statuses.
- `field_samples.npz`: shell coordinates, ECEF vectors, magnitude in microtesla.

Field lines follow the full IGRF vector in both directions with RK4, step
0.015 Earth radii and up to 2000 steps per direction. Seeds lie at radius
1.02 Earth radii; tracing stops at the surface or four Earth radii.
Truncated lines are not physically open lines. Exact geographic poles are
avoided because of the existing evaluator's pole fallback. Both branches
are joined along the magnetic-field direction; 3D magnetic arrows are omitted.

Earth-fixed coordinates and a spherical 6371.2 km reference radius are used.
IGRF describes the internal main field; solar wind and external currents are
not included. This is a static snapshot, not a magnetosphere simulation.
The 2025-2030 interval uses predicted secular variation.

`spacecraft.py` is retained for possible reuse but is not imported by this
visualization. Older `spacecraft_*` outputs, if present, are historical and
are not regenerated or displayed by the notebook.

Model reference: https://www.ncei.noaa.gov/products/international-geomagnetic-reference-field
