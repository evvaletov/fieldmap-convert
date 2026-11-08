"""Write a small analytic B/E table in metres, tesla and volts/metre."""

from itertools import product
from pathlib import Path

with Path(__file__).with_name("analytic.txt").open("w") as stream:
    stream.write("# x y z Bx By Bz Ex Ey Ez; units m, T, V/m\n")
    for x, y, z in product([0, 0.01, 0.02], [0, 0.005, 0.01], [-0.008, 0, 0.008]):
        row = (
            x,
            y,
            z,
            0.2 * x + 0.3 * z,
            -0.2 * y + 0.5 * z,
            0.7 * x - 0.1 * y + 1,
            1000 * x + 2000 * y,
            2000 * x + 3000 * y,
            -4000 * z,
        )
        stream.write(" ".join(f"{value:.17g}" for value in row) + "\n")
