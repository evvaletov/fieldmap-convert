# fieldmap-convert

Convert Cartesian electromagnetic field tables to G4beamline BLFieldMap files.
The command checks the grid, converts units, and applies explicit coordinate and
field transformations. It reads whitespace or CSV tables and COMSOL `% Grid` /
`% Data` exports. Python 3.10+ and NumPy are required.

## Install and run

```sh
python -m pip install .
fieldmap-convert input.txt output.BLFieldMap --length-unit m > conversion.json
```

The default table columns are `x y z Bx By Bz`. Coordinates must have an explicit
input unit. Magnetic fields default to tesla; electric fields default to volts
per metre. The output uses millimetres, tesla and megavolts per metre.

For a table with both fields:

```sh
fieldmap-convert input.csv output.BLFieldMap --length-unit mm \
  --delimiter comma --skiprows 1 --columns x,y,z,Bx,By,Bz,Ex,Ey,Ez
```

Column labels describe the input order. Use `_` for an ignored numeric column:

```sh
fieldmap-convert exported.txt output.BLFieldMap --length-unit m \
  --columns x,y,z,_,_,_,Bx,By,Bz
```

Blank lines and whole lines starting with `%` or `#` are ignored in table mode.
An unmarked header needs `--skiprows`. Each data row must have the declared
number of numeric columns. Decimal, `e/E`, Fortran `d/D` and Wolfram `*^`
exponents are accepted. An electric-only table uses `x,y,z,Ex,Ey,Ez`.

For COMSOL's separate coordinate axes and component blocks:

```sh
fieldmap-convert exported.txt output.BLFieldMap --format comsol-grid --length-unit m
```

This reader expects three coordinate lines after `% Grid`, then one `% Data`
block per named vector component, such as `% mf.Bx (T)`. Values run fastest
along x, then y, then z. Each supplied field needs all three components. Exports
containing repeated components from parameter sweeps must first be separated.
Field units are selected by `--b-unit` and `--e-unit`; units in component labels
are not interpreted. A `Length unit:` header must agree with `--length-unit`.

## Frames, scaling and symmetry

`--axes y,z,x` means output x is input y, output y is input z, and output z is
input x. Signed permutations are supported, for example `--axes=-x,y,z`.
For the corresponding orthogonal matrix Q, the converter applies:

```text
r_out = length_factor * Q r_in + offset_mm
E_out = electric_unit_factor * e_scale * Q E_in
B_out = magnetic_unit_factor * b_scale * det(Q) * Q B_in
```

E is a polar vector and B is an axial vector. A reflection therefore changes
magnetic signs differently from electric signs. When matching an old script
that applies the same reflected matrix to B and coordinates, account for that
extra sign explicitly with `--b-scale`; verify the intended magnet polarity.

`--offset-mm X Y Z` translates the converted coordinates. `--b-scale` and
`--e-scale` multiply the converted fields and default to 1. There is no automatic
peak normalisation or assumed magnet current. To divide by a reference field,
supply its reciprocal as the scale. Output headers set `current=1`, `gradient=1`,
`normB=1` and `normE=1`; the numeric field values already include the requested
scaling. Retain the JSON report with the map when using normalised fields.

A half-map can use an explicit G4beamline symmetry rule:

```sh
fieldmap-convert half.txt full.BLFieldMap --length-unit mm --extend y:Bx,Bz
```

The supplied y axis must start at zero. The output contains `extendY flip=Bx,Bz`;
G4beamline reflects the map at lookup time. Repeat `--extend` for another axis.
The listed signs describe a field symmetry, separately from the frame change
above. Choose them from the physical model. The converter does not infer parity.

## Validation and limits

The input must describe a uniform Cartesian grid with at least two nodes on each
axis. Rows may arrive in any order. Duplicate nodes, non-finite selected values,
nonuniform axes and missing grid nodes are rejected by default. Coordinates that differ
only by rounding are still distinct input coordinates; the tool does not merge
or round nodes. Axis spacing residuals may be at most `--grid-tolerance-mm`
(default `1e-7` mm), capped at one thousandth of the spacing.

G4beamline reads pointwise coordinates as single-precision floats. The converter
also checks that this rounding preserves each node's identity and stays within
the native 0.01 mm tolerance. A very fine grid far from the origin may fail this
check; use `--offset-mm` to centre it before validation. Increasing
`--grid-tolerance-mm` does not relax the native precision check. Field
magnitudes above the native single-precision limit are also rejected.

`--missing zero` explicitly permits absent nodes. G4beamline supplies zeros at
those locations, which can alter interpolation near holes. The report records
the count. Missing entire coordinate planes cannot be inferred from a table;
check the reported bounds and dimensions against the source export.

For exports with NaN field samples, `--nan-fields zero` explicitly replaces those
field values with zero. It records the number of replaced values and affected
nodes. Non-finite coordinates and infinite fields are always rejected. Use this option only when assuming zero field at those samples is appropriate.

The converter does not interpolate scattered samples, resample irregular grids,
combine separate E/B grids, or generate extra dimensions.

Input arrays and validation indices reside in memory; allow several times the
size of the selected float64 values. Existing output files are preserved unless
`--force` is passed. A failed read or validation does not replace an existing
map. The output path is updated atomically after writing finishes.
The input may never be used as the output, including through a hard link.

The JSON report goes to stdout and includes the input and output SHA256 hashes,
transform settings, field ranges, grid dimensions and missing-node count. Errors
go to stderr with exit status 2. Save the report if provenance is needed; source
comments and filenames are not copied into the output map.

## Example and tests

`examples/analytic.txt` is synthetic. Its generator is `examples/make_table.py`.

```sh
fieldmap-convert examples/analytic.txt analytic.BLFieldMap --length-unit m \
  --columns x,y,z,Bx,By,Bz,Ex,Ey,Ez
python -m pip install '.[test]'
python -m pytest
ruff check .
G4BL_EXECUTABLE=/path/to/g4bl python -m pytest tests/test_native.py
```

The optional native test reads the generated map with G4beamline and checks
interpolated B/E values against independent analytic formulas. The regular
suite checks all 48 signed axis permutations, units, schemas, grid coverage,
COMSOL block ordering, symmetry declarations and output preservation.

The target format is described in the
[G4beamline User's Guide, section 10.4](https://www.muonsinc.com/Website1/Muons/G4beamlineUsersGuide.pdf).
This project is an independent converter and is not part of G4beamline or COMSOL.

## Licence

MIT. The repository contains code and synthetic examples; supply your own field data.
