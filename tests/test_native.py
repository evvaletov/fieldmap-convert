"""Optional end-to-end check against an installed G4beamline executable."""

import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from fieldmap_convert.cli import main


@pytest.mark.skipif(not os.environ.get("G4BL_EXECUTABLE"), reason="set G4BL_EXECUTABLE to run")
def test_native_interpolation(tmp_path):
    executable = os.environ["G4BL_EXECUTABLE"]
    source = Path(__file__).parents[1] / "examples/analytic.txt"
    output = tmp_path / "analytic.map"
    assert (
        main(
            [str(source), str(output), "--length-unit", "m", "--columns", "x,y,z,Bx,By,Bz,Ex,Ey,Ez"]
        )
        == 0
    )
    points = np.random.default_rng(6).uniform([0, 0, -8], [20, 10, 8], (50, 3))
    np.savetxt(tmp_path / "points.txt", points, fmt="%.17g")
    (tmp_path / "probe.g4bl").write_text(
        "physics QGSP_BERT\n"
        "fieldmap analytic file=analytic.map current=1000 gradient=1000000\n"
        "box probe_volume width=100 height=100 length=100 material=Vacuum\n"
        "place probe_volume x=0 y=0 z=0\n"
        "place analytic x=0 y=0 z=0\n"
        "probefield inputFile=points.txt outputFile=fields.txt exit=1\n"
    )
    result = subprocess.run(
        [executable, "probe.g4bl"],
        cwd=tmp_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
    )
    (tmp_path / "native.log").write_text(result.stdout)
    assert result.returncode == 0, result.stdout[-5000:]
    actual = np.loadtxt(tmp_path / "fields.txt", comments="#")
    assert actual.shape == (50, 10)
    x, y, z = (points / 1000).T
    expected = np.column_stack(
        (
            0.2 * x + 0.3 * z,
            -0.2 * y + 0.5 * z,
            0.7 * x - 0.1 * y + 1,
            (1000 * x + 2000 * y) / 1e6,
            (2000 * x + 3000 * y) / 1e6,
            -4000 * z / 1e6,
        )
    )
    # probefield prints six decimals; magnification resolves small field components.
    actual[:, 4:7] /= 1000
    actual[:, 7:] /= 1e6
    np.testing.assert_allclose(actual[:, 4:], expected, rtol=2e-7, atol=6e-10)
