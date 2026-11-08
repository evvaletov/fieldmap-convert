"""Readers, coordinate transforms and validated BLFieldMap output."""

from dataclasses import dataclass
import hashlib
from itertools import chain
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np

from . import __version__

LENGTH_UNITS = {"mm": 1.0, "cm": 10.0, "m": 1000.0}
B_UNITS = {"T": 1.0, "mT": 0.001, "G": 0.0001}
E_UNITS = {"V/m": 0.000001, "kV/m": 0.001, "MV/m": 1.0}
NAMES = ("x", "y", "z", "Bx", "By", "Bz", "Ex", "Ey", "Ez")
CANONICAL = {name.lower(): name for name in NAMES}
NATIVE_POINT_TOLERANCE_MM = 0.01


@dataclass
class FieldMap:
    """Rows in x,y,z,Bx,By,Bz,Ex,Ey,Ez order; absent field triples are zero."""

    rows: np.ndarray
    components: tuple[str, ...]
    source_sha256: str
    declared_length_unit: str | None = None
    nan_values_replaced: int = 0
    nan_nodes_repaired: int = 0


class Source:
    """Hash the same bytes that are decoded and consumed by a reader."""

    def __init__(self, path):
        self.path = Path(path)
        self.digest = hashlib.sha256()
        self.declared_length_unit = None

    def lines(self):
        with self.path.open("rb") as stream:
            for number, raw in enumerate(stream, 1):
                self.digest.update(raw)
                line = raw.decode("utf-8-sig" if number == 1 else "utf-8")
                if line.lstrip().startswith("%"):
                    match = re.match(r"\s*%\s*Length unit:\s*(\S+)", line, re.I)
                    if match:
                        self.declared_length_unit = match[1]
                yield line


def component_names(names):
    """Validate and order complete B and/or E component triples."""
    if len(set(names)) != len(names):
        raise ValueError("duplicate field components")
    chosen = set(names)
    if not chosen or chosen - set(NAMES[3:]):
        raise ValueError("field components must be Bx,By,Bz and/or Ex,Ey,Ez")
    for triple in (set(NAMES[3:6]), set(NAMES[6:])):
        if chosen & triple and not triple <= chosen:
            raise ValueError("each supplied field must have all three vector components")
    return tuple(name for name in NAMES[3:] if name in chosen)


def numbers(line):
    """Accept decimal, e/E, Fortran d/D and Wolfram *^ exponents."""
    return line.replace("D", "e").replace("d", "e").replace("*^", "e")


def read_table(path, columns, *, skiprows=0, delimiter=None, nan_fields="error"):
    """Read a numeric table with an explicit label for every column (_ ignores one)."""
    if skiprows < 0:
        raise ValueError("skiprows must be non-negative")
    labels = [CANONICAL.get(x.strip().lower(), x.strip()) for x in columns]
    if any(x != "_" and x not in NAMES for x in labels):
        raise ValueError("unknown column name; use _ for an ignored numeric column")
    selected = [x for x in labels if x != "_"]
    if len(set(selected)) != len(selected) or not set(NAMES[:3]) <= set(selected):
        raise ValueError("columns must contain x,y,z exactly once")
    components = component_names([x for x in selected if x not in NAMES[:3]])
    source = Source(path)

    def data_lines():
        for number, line in enumerate(source.lines()):
            if number < skiprows or not line.strip() or line.lstrip().startswith(("#", "%")):
                continue
            yield numbers(line)

    lines = iter(data_lines())
    first = next(lines, None)
    if first is None:
        raise ValueError("empty table")
    table = np.loadtxt(chain([first], lines), delimiter=delimiter, comments="#", ndmin=2)
    if table.shape[1] != len(labels):
        raise ValueError("column count differs from --columns")
    rows = np.zeros((len(table), 9), dtype=np.float64)
    for index, name in enumerate(labels):
        if name != "_":
            rows[:, NAMES.index(name)] = table[:, index]
    del table
    return finish_read(rows, components, source, nan_fields)


def read_comsol_grid(path, *, nan_fields="error"):
    """Read COMSOL % Grid / % Data exports, with x varying fastest in each block."""
    source = Source(path)
    lines = iter(source.lines())
    for line in lines:
        if line.strip().lower() == "% grid":
            break
    else:
        raise ValueError("missing COMSOL % Grid marker")
    axes = []
    for line in lines:
        if not line.strip():
            continue
        if line.lstrip().startswith("%"):
            raise ValueError("expected three coordinate lines after % Grid")
        axes.append(np.array([float(x) for x in numbers(line).split()]))
        if len(axes) == 3:
            break
    if len(axes) != 3 or any(len(axis) < 2 for axis in axes):
        raise ValueError("COMSOL grid needs at least two nodes on each axis")
    count = math.prod(len(axis) for axis in axes)
    if count > np.iinfo(np.int32).max:
        raise ValueError("grid exceeds the native 32-bit node-index range")
    rows = np.zeros((count, 9), dtype=np.float64)
    seen = []
    pending = None
    cursor = 0
    expect_label = False

    def finish_block():
        if pending is None:
            return
        if cursor != count:
            raise ValueError(f"{pending}: expected {count} field values, found {cursor}")

    for line in lines:
        text = line.strip()
        if not text:
            continue
        if text.lower() == "% data":
            if expect_label:
                raise ValueError("missing field label after % Data")
            finish_block()
            cursor = 0
            pending = None
            expect_label = True
        elif expect_label:
            match = re.match(r"%\s*(?:[A-Za-z_][\w]*\.)?([BE][xyz])\b", text, re.I)
            if not match:
                raise ValueError("expected a Bx/By/Bz/Ex/Ey/Ez label after % Data")
            pending = CANONICAL[match[1].lower()]
            if pending in seen:
                raise ValueError(
                    f"duplicate COMSOL component {pending}; select one parameter export"
                )
            seen.append(pending)
            expect_label = False
        elif text.startswith("%"):
            raise ValueError("unexpected comment inside COMSOL field block")
        elif pending is None:
            raise ValueError("numeric field data without a % Data component label")
        else:
            values = [float(x) for x in numbers(text).split()]
            end = cursor + len(values)
            if end > count:
                raise ValueError(f"{pending}: expected {count} field values, found at least {end}")
            rows[cursor:end, NAMES.index(pending)] = values
            cursor = end
    if expect_label:
        raise ValueError("missing final COMSOL component label")
    finish_block()
    components = component_names(seen)
    nx, ny, nz = (len(axis) for axis in axes)
    rows[:, 0] = np.tile(axes[0], ny * nz)
    rows[:, 1] = np.tile(np.repeat(axes[1], nx), nz)
    rows[:, 2] = np.repeat(axes[2], nx * ny)
    return finish_read(rows, components, source, nan_fields)


def finish_read(rows, components, source, nan_fields):
    if nan_fields not in ("error", "zero"):
        raise ValueError("nan_fields must be error or zero")
    count = nodes = 0
    if nan_fields == "zero":
        mask = np.isnan(rows[:, 3:])
        count = int(mask.sum())
        if count:
            nodes = int(np.count_nonzero(np.any(mask, axis=1)))
            rows[:, 3:][mask] = 0.0
    require_finite(rows)
    return FieldMap(
        rows,
        components,
        source.digest.hexdigest(),
        source.declared_length_unit,
        count,
        nodes,
    )


def require_finite(rows):
    if rows.ndim != 2 or rows.shape[1] != 9 or not len(rows):
        raise ValueError("expected a non-empty array with nine columns")
    if not np.isfinite(rows).all():
        raise ValueError("non-finite coordinate or field value")


def frame_matrix(axes):
    """Build Q for output coordinates = Q @ input coordinates."""
    if len(axes) != 3:
        raise ValueError("axes must contain three signed input axes, e.g. y,z,x")
    matrix = np.zeros((3, 3))
    used = set()
    for index, token in enumerate(axes):
        match = re.fullmatch(r"([+-]?)([xyz])", token.strip().lower())
        if not match or match[2] in used:
            raise ValueError("axes must be a signed permutation of x,y,z")
        used.add(match[2])
        matrix[index, "xyz".index(match[2])] = -1 if match[1] == "-" else 1
    return matrix


def transform(
    field,
    *,
    length_unit,
    b_unit="T",
    e_unit="V/m",
    axes=("x", "y", "z"),
    offset_mm=(0.0, 0.0, 0.0),
    b_scale=1.0,
    e_scale=1.0,
):
    """Transform coordinates, polar E and axial B; return a new map in mm,T,MV/m."""
    if length_unit not in LENGTH_UNITS or b_unit not in B_UNITS or e_unit not in E_UNITS:
        raise ValueError("unsupported unit")
    if field.declared_length_unit and field.declared_length_unit != length_unit:
        raise ValueError("--length-unit conflicts with the source's Length unit header")
    offset = np.asarray(offset_mm, dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("offset must contain three finite values in output millimetres")
    if not np.isfinite([b_scale, e_scale]).all():
        raise ValueError("field scale factors must be finite")
    require_finite(field.rows)
    q = frame_matrix(axes)
    determinant = round(np.linalg.det(q))
    result = np.empty_like(field.rows)
    with np.errstate(over="ignore", invalid="ignore"):
        for start, factor in (
            (0, LENGTH_UNITS[length_unit]),
            (3, determinant * B_UNITS[b_unit] * b_scale),
            (6, E_UNITS[e_unit] * e_scale),
        ):
            target = result[:, start : start + 3]
            np.matmul(field.rows[:, start : start + 3], q.T, out=target)
            target *= factor
        result[:, :3] += offset
    require_finite(result)
    return FieldMap(
        result,
        field.components,
        field.source_sha256,
        nan_values_replaced=field.nan_values_replaced,
        nan_nodes_repaired=field.nan_nodes_repaired,
    )


def validate_grid(field, *, tolerance_mm=1e-7, missing="error"):
    """Check uniform axes, unique nodes and Cartesian coverage without a dense grid."""
    require_finite(field.rows)
    maximum_field = max(float(field.rows[:, 3:].max()), -float(field.rows[:, 3:].min()))
    if maximum_field > float(np.finfo(np.float32).max):
        raise ValueError("field value exceeds the native single-precision range")
    if not math.isfinite(tolerance_mm) or tolerance_mm < 0:
        raise ValueError("grid tolerance must be finite and non-negative")
    if missing not in ("error", "zero"):
        raise ValueError("missing must be error or zero")
    origin, spacing, shape = [], [], []
    linear = np.zeros(len(field.rows), dtype=np.int64)
    stride = 1
    max_residual = 0.0
    max_native_residual = 0.0
    for column in range(3):
        axis, inverse = np.unique(field.rows[:, column], return_inverse=True)
        if len(axis) < 2:
            raise ValueError("3-D maps need at least two distinct nodes on each axis")
        step = (axis[-1] - axis[0]) / (len(axis) - 1)
        if not math.isfinite(step) or step <= 0:
            raise ValueError("invalid grid spacing")
        regular = axis[0] + np.arange(len(axis)) * step
        residual = float(np.max(np.abs(axis - regular)))
        # A large tolerance must not make distinct lattice nodes indistinguishable.
        if residual > min(tolerance_mm, step * 0.001):
            raise ValueError(f"non-uniform {'xyz'[column]} axis; residual {residual:g} mm")
        # The native pointwise reader parses coordinates as single-precision floats.
        with np.errstate(over="ignore", invalid="ignore"):
            native_axis = axis.astype(np.float32).astype(np.float64)
        native_residual = float(np.max(np.abs(native_axis - regular)))
        if native_residual > min(NATIVE_POINT_TOLERANCE_MM, step * 0.49):
            raise ValueError(
                "grid cannot retain its nodes within native pointwise precision; "
                "centre the map with --offset-mm or increase its node spacing"
            )
        max_native_residual = max(max_native_residual, native_residual)
        origin.append(float(axis[0]))
        spacing.append(float(step))
        shape.append(len(axis))
        if stride * len(axis) > np.iinfo(np.int32).max:
            raise ValueError("grid exceeds the native 32-bit node-index range")
        linear += inverse * stride
        stride *= len(axis)
        max_residual = max(max_residual, residual)
    linear.sort()
    if np.any(linear[1:] == linear[:-1]):
        raise ValueError("duplicate grid nodes")
    missing_count = stride - len(field.rows)
    if missing_count and missing == "error":
        raise ValueError(f"incomplete Cartesian grid: {missing_count} missing nodes")
    return {
        "origin_mm": origin,
        "spacing_mm": spacing,
        "shape": shape,
        "input_nodes": len(field.rows),
        "grid_nodes": stride,
        "missing_nodes": missing_count,
        "missing_policy": missing,
        "max_axis_residual_mm": max_residual,
        "max_native_axis_residual_mm": max_native_residual,
    }


def parse_extensions(specs, grid, components):
    """Validate explicit reflection parities around an axis starting at zero."""
    extensions = {}
    for spec in specs:
        axis, sep, flips = spec.partition(":")
        if not sep or axis not in "xyz" or len(axis) != 1 or axis in extensions:
            raise ValueError("extensions must be unique axis:components entries, e.g. y:Bx,Bz")
        names = [] if not flips else [CANONICAL.get(x.strip().lower()) for x in flips.split(",")]
        if len(set(names)) != len(names) or any(name not in components for name in names):
            raise ValueError("extension flips must name distinct, supplied field components")
        if grid["origin_mm"]["xyz".index(axis)] != 0.0:
            raise ValueError("an extended axis must start at exactly zero")
        extensions[axis] = names
    return extensions


def write_map(path, field, grid, *, extensions=None, force=False):
    """Write a file atomically and return its hash; replacement requires force."""
    path = Path(path)
    extensions = extensions or {}
    headers = [
        f"# fieldmap-convert {__version__}; coordinates mm; B tesla; E megavolts/metre",
        f"# source SHA256 {field.source_sha256}",
        f"# NaN field values replaced with zero: {field.nan_values_replaced}",
        f"# missing nodes: {grid['missing_nodes']}; policy: {grid['missing_policy']}",
        "param maxline=1024 current=1 gradient=1 normB=1 normE=1",
    ]
    args = [f"{axis.upper()}0={value:.17g}" for axis, value in zip("xyz", grid["origin_mm"])]
    args += [f"n{axis.upper()}={value}" for axis, value in zip("xyz", grid["shape"])]
    args += [f"d{axis.upper()}={value:.17g}" for axis, value in zip("xyz", grid["spacing_mm"])]
    args.append(f"tolerance={NATIVE_POINT_TOLERANCE_MM:.17g}")
    headers.append("grid " + " ".join(args))
    for axis, names in extensions.items():
        headers.append(f"extend{axis.upper()}" + (" flip=" + ",".join(names) if names else ""))
    headers.append("data")
    width = 9 if "Ex" in field.components else 6
    temp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="ascii", dir=path.parent, prefix=".fieldmap-", delete=False
        ) as stream:
            temp = Path(stream.name)
            stream.write("\n".join(headers) + "\n")
            np.savetxt(stream, field.rows[:, :width], fmt="%.17g")
            stream.flush()
            os.fsync(stream.fileno())
        digest = sha256_file(temp)
        if force:
            os.replace(temp, path)
        else:
            os.link(temp, path)
        return digest
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
