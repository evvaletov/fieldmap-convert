import hashlib
import itertools
import json
import os

import numpy as np
import pytest

from fieldmap_convert.cli import main
from fieldmap_convert.core import (
    FieldMap,
    frame_matrix,
    parse_extensions,
    read_comsol_grid,
    read_table,
    transform,
    validate_grid,
    write_map,
)


@pytest.fixture
def field():
    xyz = np.array(list(itertools.product([0.0, 10.0, 20.0], [0.0, 5.0, 10.0], [-8.0, 0.0, 8.0])))
    x, y, z = xyz.T
    b = np.column_stack((2 * x + 3 * z, -2 * y + 5 * z, 7 * x - y + 1))
    e = np.column_stack((x + 2 * y, 2 * x + 3 * y, -4 * z))
    return FieldMap(np.column_stack((xyz, b, e)), ("Bx", "By", "Bz", "Ex", "Ey", "Ez"), "a" * 64)


def output_rows(path):
    lines = path.read_text().splitlines()
    return np.loadtxt(lines[lines.index("data") + 1 :], ndmin=2)


def test_units_and_source_hash(tmp_path, field):
    path = tmp_path / "source.txt"
    source = field.rows.copy()
    source[:, :3] /= 1000
    source[:, 3:6] *= 1000
    source[:, 6:] *= 1e6
    np.savetxt(path, source)
    loaded = read_table(path, "x y z Bx By Bz Ex Ey Ez".split())
    assert loaded.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    converted = transform(loaded, length_unit="m", b_unit="mT", e_unit="V/m")
    np.testing.assert_allclose(converted.rows, field.rows, atol=1e-13)


def test_permutation_and_offset(field):
    result = transform(
        field,
        length_unit="cm",
        axes=["y", "z", "x"],
        offset_mm=[1, 2, 3],
        b_scale=-2,
        e_unit="MV/m",
        e_scale=3,
    )
    np.testing.assert_allclose(result.rows[:, :3], field.rows[:, [1, 2, 0]] * 10 + [1, 2, 3])
    np.testing.assert_allclose(result.rows[:, 3:6], field.rows[:, [4, 5, 3]] * -2)
    np.testing.assert_allclose(result.rows[:, 6:], field.rows[:, [7, 8, 6]] * 3)


def test_improper_frame_preserves_cross_product():
    # B transforms like the cross product of two polar vectors.
    u, v = np.array([2.0, 3.0, 5.0]), np.array([7.0, 11.0, 13.0])
    b = np.cross(u, v)
    rows = np.array([[1, 2, 3, *b, *u]], dtype=float)
    field = FieldMap(rows, ("Bx", "By", "Bz", "Ex", "Ey", "Ez"), "b" * 64)
    for permutation in itertools.permutations("xyz"):
        for signs in itertools.product(["", "-"], repeat=3):
            axes = [s + a for s, a in zip(signs, permutation)]
            q = frame_matrix(axes)
            actual = transform(field, length_unit="mm", e_unit="MV/m", axes=axes)
            np.testing.assert_allclose(actual.rows[0, 3:6], np.cross(q @ u, q @ v))
            np.testing.assert_allclose(actual.rows[0, 6:], q @ u)


def test_shuffled_grid_and_written_values(tmp_path, field):
    np.random.default_rng(3).shuffle(field.rows)
    grid = validate_grid(field)
    assert grid["shape"] == [3, 3, 3]
    assert grid["spacing_mm"] == [10, 5, 8]
    path = tmp_path / "map.BLFieldMap"
    write_map(path, field, grid)
    np.testing.assert_array_equal(output_rows(path), field.rows)
    assert "normE=1" in path.read_text()


def test_grid_errors(field):
    with pytest.raises(ValueError, match="duplicate"):
        validate_grid(FieldMap(np.vstack((field.rows, field.rows[0])), field.components, ""))
    sparse = FieldMap(field.rows[1:].copy(), field.components, "")
    with pytest.raises(ValueError, match="1 missing"):
        validate_grid(sparse)
    assert validate_grid(sparse, missing="zero")["missing_nodes"] == 1
    field.rows[field.rows[:, 0] == 10, 0] = 11
    with pytest.raises(ValueError, match="non-uniform"):
        validate_grid(field)
    with pytest.raises(ValueError, match="non-uniform"):
        validate_grid(field, tolerance_mm=1000)


def test_roundoff_is_bounded(field):
    field.rows[field.rows[:, 0] == 10, 0] += 1e-9
    assert 0 < validate_grid(field)["max_axis_residual_mm"] < 1e-7


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_rejected(tmp_path, field, value):
    field.rows[0, 4] = value
    path = tmp_path / "bad.txt"
    np.savetxt(path, field.rows)
    with pytest.raises(ValueError, match="non-finite"):
        read_table(path, "x y z Bx By Bz Ex Ey Ez".split())


@pytest.mark.parametrize(
    "columns", ["x,y,z,Bx", "x,y,z,Bx,Bx,Bz", "x,y,z,a,b,c", "x,y,y,Bx,By,Bz", "x,y,z,_,_,_"]
)
def test_invalid_schema(tmp_path, columns):
    p = tmp_path / "a.txt"
    p.write_text("0 0 0 1 2 3\n")
    with pytest.raises(ValueError):
        read_table(p, columns.split(","))


def test_comsol_table_and_exponents(tmp_path):
    path = tmp_path / "table.txt"
    text = "\ufeff% Length unit: m\n% model comment\n1D-3 2d-3 3*^-3 99 4 5 6\n"
    path.write_text(text)
    field = read_table(path, "x y z _ Bx By Bz".split())
    np.testing.assert_allclose(field.rows[0, :6], [0.001, 0.002, 0.003, 4, 5, 6])
    assert field.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="conflicts"):
        transform(field, length_unit="mm")


def comsol_text():
    axes = [[0, 2, 4], [-1, 1], [3, 6]]
    lines = ["% Length unit: mm", "% Grid"] + [" ".join(map(str, a)) for a in axes]
    # Asymmetric analytic values expose wrong x/y/z flattening orders.
    for name, function in [
        ("Bx", lambda x, y, z: x + 10 * y + 100 * z),
        ("By", lambda x, y, z: 7 * x - 3 * y + z),
        ("Bz", lambda x, y, z: -x + y - 5 * z),
    ]:
        lines += ["% Data", "% mf." + name + " (T) @ parameter=1"]
        for z in axes[2]:
            for y in axes[1]:
                lines.append(" ".join(str(function(x, y, z)) for x in axes[0]))
    return "\n".join(lines) + "\n"


def test_comsol_grid_order_and_hash(tmp_path):
    path = tmp_path / "grid.txt"
    path.write_text(comsol_text())
    field = read_comsol_grid(path)
    x, y, z = field.rows[:, :3].T
    np.testing.assert_allclose(field.rows[:, 3], x + 10 * y + 100 * z)
    np.testing.assert_allclose(field.rows[:, 4], 7 * x - 3 * y + z)
    np.testing.assert_allclose(field.rows[:, 5], -x + y - 5 * z)
    assert validate_grid(field)["shape"] == [3, 2, 2]
    assert field.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("change", ["missing", "excess", "duplicate", "unknown", "trailing", "nan"])
def test_malformed_comsol(tmp_path, change):
    text = comsol_text()
    if change == "missing":
        text = text.rsplit("\n", 2)[0] + "\n"
    if change == "excess":
        text += "1 2 3\n"
    if change == "duplicate":
        text = text.replace("mf.Bz", "mf.By")
    if change == "unknown":
        text = text.replace("mf.Bz", "mf.normB")
    if change == "trailing":
        text += "% Data\n"
    if change == "nan":
        text = text.replace("290 292 294", "nan 292 294")
    path = tmp_path / "bad.txt"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_comsol_grid(path)


def test_extensions_and_electric_only(tmp_path, field):
    grid = validate_grid(field)
    assert parse_extensions(["y:Bx,Bz"], grid, field.components) == {"y": ["Bx", "Bz"]}
    for spec in ["z:Bx", "xy:Bx", "y:missing", "y:Bx,Bx"]:
        with pytest.raises(ValueError):
            parse_extensions([spec], grid, field.components)
    with pytest.raises(ValueError):
        parse_extensions(["y:", "y:"], grid, field.components)
    field.rows[:, 3:6] = 0
    field.components = ("Ex", "Ey", "Ez")
    p = tmp_path / "electric.map"
    write_map(p, field, grid)
    assert output_rows(p).shape == (27, 9)


def test_atomic_overwrite_and_input_protection(tmp_path, field, capsys):
    src = tmp_path / "source.txt"
    dst = tmp_path / "output.map"
    np.savetxt(src, field.rows[:, :6])
    dst.write_text("existing")
    args = [str(src), str(dst), "--length-unit", "mm"]
    assert main(args) == 2 and dst.read_text() == "existing"
    assert main(args + ["--force"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["grid"]["grid_nodes"] == 27
    assert report["output_sha256"] == hashlib.sha256(dst.read_bytes()).hexdigest()
    original = src.read_bytes()
    assert main([str(src), str(src), "--length-unit", "mm", "--force"]) == 2
    assert src.read_bytes() == original
    linked = tmp_path / "link.txt"
    os.link(src, linked)
    assert main([str(src), str(linked), "--length-unit", "mm", "--force"]) == 2
    assert src.read_bytes() == original
    src.write_text("0 0 0 nan 0 0\n")
    saved = dst.read_bytes()
    assert main(args + ["--force"]) == 2 and dst.read_bytes() == saved
    assert not list(tmp_path.glob(".fieldmap-*"))


def test_empty_singleton_and_csv(tmp_path):
    path = tmp_path / "empty"
    path.write_text("% comment\n")
    with pytest.raises(ValueError, match="empty table"):
        read_table(path, "x y z Bx By Bz".split())
    path.write_text("header\n1,2,3,4,5,6\n")
    field = read_table(path, "x y z Bx By Bz".split(), skiprows=1, delimiter=",")
    with pytest.raises(ValueError, match="two distinct"):
        validate_grid(field)


def test_explicit_nan_field_policy(tmp_path, field, capsys):
    field.rows[0, 3] = float("nan")
    field.rows[0, 4] = float("nan")
    field.rows[1, 8] = float("nan")
    src = tmp_path / "source.txt"
    dst = tmp_path / "map"
    np.savetxt(src, field.rows)
    args = [
        str(src),
        str(dst),
        "--length-unit",
        "mm",
        "--e-unit",
        "MV/m",
        "--columns",
        "x,y,z,Bx,By,Bz,Ex,Ey,Ez",
    ]
    assert main(args) == 2 and not dst.exists()
    assert main(args + ["--nan-fields", "zero"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["nan_values_replaced"] == 3
    assert report["nan_nodes_repaired"] == 2
    np.testing.assert_array_equal(output_rows(dst), np.nan_to_num(field.rows))
    field.rows[0, 0] = float("nan")
    np.savetxt(src, field.rows)
    with pytest.raises(ValueError, match="non-finite"):
        read_table(src, "x y z Bx By Bz Ex Ey Ez".split(), nan_fields="zero")
    field.rows[0, 0] = 0
    field.rows[0, 3] = float("inf")
    np.savetxt(src, field.rows)
    with pytest.raises(ValueError, match="non-finite"):
        read_table(src, "x y z Bx By Bz Ex Ey Ez".split(), nan_fields="zero")


def test_broken_output_symlink_fails_before_read(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source"
    source.write_text("invalid data")
    output = tmp_path / "output"
    output.symlink_to(tmp_path / "missing")

    def unexpected_read(*args, **kwargs):
        pytest.fail("existing output must be checked before loading the source")

    monkeypatch.setattr("fieldmap_convert.cli.read_table", unexpected_read)
    assert main([str(source), str(output), "--length-unit", "mm"]) == 2
    assert "output exists" in capsys.readouterr().err
    assert output.is_symlink()


def test_native_coordinate_precision_and_recentering(field):
    field.rows[:, 0] = 1000 + field.rows[:, 0] * 1e-7
    with pytest.raises(ValueError, match="native pointwise precision"):
        validate_grid(field)
    recentered = transform(field, length_unit="mm", offset_mm=[-1000, 0, 0])
    assert validate_grid(recentered)["max_native_axis_residual_mm"] < 1e-7


def test_large_tolerance_cannot_exceed_native_reader(field):
    field.rows[:, 0] *= 100
    field.rows[field.rows[:, 0] == 1000, 0] += 0.1
    with pytest.raises(ValueError, match="native pointwise precision"):
        validate_grid(field, tolerance_mm=0.2)


@pytest.mark.parametrize("column,value", [(3, 1e40), (6, -1e40)])
def test_native_field_range(field, column, value):
    field.rows[0, column] = value
    with pytest.raises(ValueError, match="native single-precision range"):
        validate_grid(field)
