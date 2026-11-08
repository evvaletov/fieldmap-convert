"""Command-line entry point."""

import argparse
import json
import os
from pathlib import Path
import sys

from . import __version__
from .core import (
    B_UNITS,
    E_UNITS,
    LENGTH_UNITS,
    NAMES,
    parse_extensions,
    read_comsol_grid,
    read_table,
    transform,
    validate_grid,
    write_map,
)


def parser():
    result = argparse.ArgumentParser(
        description="Convert Cartesian field data to a validated G4beamline BLFieldMap."
    )
    result.add_argument("input", type=Path)
    result.add_argument("output", type=Path)
    result.add_argument("--version", action="version", version=__version__)
    result.add_argument("--format", choices=["table", "comsol-grid"], default="table")
    result.add_argument(
        "--columns",
        default="x,y,z,Bx,By,Bz",
        help="table column labels in order; _ ignores a numeric column",
    )
    result.add_argument("--skiprows", type=int, default=0, help="leading table lines to skip")
    result.add_argument("--delimiter", choices=["whitespace", "comma"], default="whitespace")
    result.add_argument("--length-unit", choices=LENGTH_UNITS, required=True)
    result.add_argument("--b-unit", choices=B_UNITS, default="T")
    result.add_argument("--e-unit", choices=E_UNITS, default="V/m")
    result.add_argument(
        "--axes", default="x,y,z", help="output x,y,z as signed input axes (use --axes=-x,y,z)"
    )
    result.add_argument(
        "--offset-mm",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
        help="translation after the frame change",
    )
    result.add_argument("--b-scale", type=float, default=1.0)
    result.add_argument("--e-scale", type=float, default=1.0)
    result.add_argument("--grid-tolerance-mm", type=float, default=1e-7)
    result.add_argument(
        "--missing",
        choices=["error", "zero"],
        default="error",
        help="missing Cartesian nodes: reject (default) or explicitly use zero",
    )
    result.add_argument(
        "--extend",
        action="append",
        default=[],
        metavar="AXIS:FLIPS",
        help="reflect a half-map, e.g. y:Bx,Bz (repeat for other axes)",
    )
    result.add_argument(
        "--force", action="store_true", help="replace an existing output atomically"
    )
    result.add_argument(
        "--nan-fields",
        choices=["error", "zero"],
        default="error",
        help="reject NaN fields (default) or explicitly replace them with zero",
    )
    return result


def run(args):
    if args.input.resolve() == args.output.resolve():
        raise ValueError("input and output must be different files")
    if args.output.exists() and os.path.samefile(args.input, args.output):
        raise ValueError("output is a hard link to the input")
    if os.path.lexists(args.output) and not args.force:
        raise ValueError("output exists; choose another name or pass --force")
    if args.format == "table":
        field = read_table(
            args.input,
            args.columns.split(","),
            skiprows=args.skiprows,
            delimiter="," if args.delimiter == "comma" else None,
            nan_fields=args.nan_fields,
        )
    else:
        if args.skiprows or args.delimiter != "whitespace" or args.columns != "x,y,z,Bx,By,Bz":
            raise ValueError("--skiprows, --delimiter and --columns apply only to table input")
        field = read_comsol_grid(args.input, nan_fields=args.nan_fields)
    settings = {
        "length_unit": args.length_unit,
        "b_unit": args.b_unit,
        "e_unit": args.e_unit,
        "axes": args.axes.split(","),
        "offset_mm": args.offset_mm,
        "b_scale": args.b_scale,
        "e_scale": args.e_scale,
    }
    field = transform(field, **settings)
    grid = validate_grid(field, tolerance_mm=args.grid_tolerance_mm, missing=args.missing)
    extensions = parse_extensions(args.extend, grid, field.components)
    output_hash = write_map(args.output, field, grid, extensions=extensions, force=args.force)
    ranges = {
        name: [
            float(field.rows[:, NAMES.index(name)].min()),
            float(field.rows[:, NAMES.index(name)].max()),
        ]
        for name in field.components
    }
    return {
        "version": __version__,
        "input_sha256": field.source_sha256,
        "nan_field_policy": args.nan_fields,
        "nan_values_replaced": field.nan_values_replaced,
        "nan_nodes_repaired": field.nan_nodes_repaired,
        "output_sha256": output_hash,
        "settings": settings,
        "grid": grid,
        "extensions": extensions,
        "field_ranges_T_or_MVm": ranges,
    }


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        report = run(args)
    except (OSError, ValueError) as error:
        print(f"fieldmap-convert: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0
