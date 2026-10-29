
#!/usr/bin/env python3
"""SSFP natural error propagation with exact encoded-input references.

Each Noir program is compiled manually. This runner updates Prover.toml,
executes the existing circuit, and writes summary.csv.

Usage:
    python run_propagation.py
    python run_propagation.py --omit-sum
"""

import argparse
import csv
import hashlib
import random
import re
import subprocess
from pathlib import Path

from mpmath import mp


# Experiment settings
N_VALUES = (10, 100, 1000, 10000)
REPETITIONS = {10: 50, 100: 200, 1000: 200, 10000: 200}
D_VALUES = (8, 16)
INITIAL_LEVEL = 1
MP_DECIMAL_DIGITS = 100
RANDOM_SEED = 20261025
NARGO = "nargo"

ROOT = Path(__file__).resolve().parent

FIELD_MODULUS = (
    21888242871839275222246405745257275088548364400416034343698204186575808495617
)

RESULT_RE = re.compile(
    r"SSFP_PROP\s+(8|16)\s+([01])\s+"
    r"(0[xX][0-9a-fA-F]+|\d+)\s+"
    r"(0[xX][0-9a-fA-F]+|\d+)\s*$"
)


def create_inputs(workload, n, trial):
    """Generate encoded inputs and an independent reference for each D."""

    seed = int.from_bytes(
        hashlib.sha256(
            f"{RANDOM_SEED}|{workload}|{trial}".encode()
        ).digest(),
        "big",
    )
    rng = random.Random(seed)

    values = {d: ([], []) for d in D_VALUES}
    references = {}

    if workload == "sum":
        # Common 48-bit random values, quantized separately at D=8 and D=16.
        # Products are exact at level 2; no explicit reset is requested.

        for _ in range(n):
            x = rng.randint(-(1 << 47), 1 << 47)
            y = rng.randint(-(1 << 47), 1 << 47)

            for d in D_VALUES:
                a, b = values[d]

                a.append(
                    (1 if x >= 0 else -1) * (abs(x) >> (48 - d))
                )
                b.append(
                    (1 if y >= 0 else -1) * (abs(y) >> (48 - d))
                )

        for d in D_VALUES:
            a, b = values[d]
            references[d] = (
                mp.mpf(sum(x * y for x, y in zip(a, b)))
                / (1 << (2 * d))
            )

    else:
        # Product of sums.
        # Consecutive factors are 1+delta and 1-delta.
        # This keeps the exact product bounded even for very large N.

        assert n % 2 == 0

        reference = mp.mpf(1)

        for _ in range(n // 2):
            delta = rng.choice((1, 2))
            sign = rng.choice((-1, 1))

            for d in D_VALUES:
                scale = 1 << d
                step = delta * (scale // 256)
                a, b = values[d]

                a.extend((scale // 2, scale // 2))
                b.extend((
                    scale // 2 + sign * step,
                    scale // 2 - sign * step,
                ))

            reference *= 1 - (mp.mpf(delta) / 256) ** 2

        references = {d: reference for d in D_VALUES}

    return values, references


def field_encode(mantissa, d):
    return (
        mantissa << (d * (INITIAL_LEVEL - 1))
    ) % FIELD_MODULUS


def write_prover(project, values):
    def toml_array(numbers):
        return "[" + ", ".join(f'"0x{x:x}"' for x in numbers) + "]"

    content = "".join(
        f"{name} = {toml_array([field_encode(x, d) for x in sequence])}\n"
        for d in D_VALUES
        for name, sequence in (
            (f"a{d}", values[d][0]),
            (f"b{d}", values[d][1]),
        )
    )

    (project / "Prover.toml").write_text(
        content,
        encoding="utf-8",
    )


def execute(project, workload):
    completed = subprocess.run(
        [NARGO, "execute"],
        cwd=project,
        capture_output=True,
        text=True,
    )

    if completed.returncode:
        raise RuntimeError(
            f"nargo execute failed in {project.name}:\n"
            + (completed.stdout + completed.stderr)[-5000:]
        )

    found = {}

    for line in (completed.stdout + "\n" + completed.stderr).splitlines():
        if "SSFP_PROP" not in line:
            continue

        match = RESULT_RE.search(line.strip())

        if match is None:
            raise RuntimeError(f"Unrecognized result line: {line}")

        d, rounding, raw, level = (
            int(item, 0) for item in match.groups()
        )

        key = (d, rounding)

        if key in found or not (
            0 <= raw < FIELD_MODULUS and 0 <= level <= 64
        ):
            raise RuntimeError(
                f"Invalid or duplicate result line: {line}"
            )

        signed = (
            raw if raw < FIELD_MODULUS // 2
            else raw - FIELD_MODULUS
        )

        found[key] = (
            mp.mpf(signed) / (1 << (d * level))
        )

    modes = (0,) if workload == "sum" else (0, 1)
    expected = {
        (d, r)
        for d in D_VALUES
        for r in modes
    }

    if set(found) != expected:
        raise RuntimeError(
            f"Expected {sorted(expected)} in {project.name}; "
            f"got {sorted(found)}"
        )

    return found


def percentile95(sorted_errors):
    pos = mp.mpf("0.95") * (len(sorted_errors) - 1)
    low = int(mp.floor(pos))
    high = int(mp.ceil(pos))

    return (
        sorted_errors[low]
        + (sorted_errors[high] - sorted_errors[low]) * (pos - low)
    )


def summarize(
    workload,
    n,
    d,
    rounding,
    signed_errors,
    relative_errors,
):
    absolute = sorted(abs(error) for error in signed_errors)

    fmt = lambda value: mp.nstr(value, 14)

    return {
        "workload": (
            "sum_of_products"
            if workload == "sum"
            else "paired_product_of_sums"
        ),
        "N": n,
        "D": d,
        "L": INITIAL_LEVEL,
        "rounding": (
            "N/A" if workload == "sum" else f"R{rounding}"
        ),
        "RR": "" if workload == "sum" else rounding,
        "IR": "" if workload == "sum" else rounding,
        "samples": len(absolute),
        "valid": len(absolute),
        "failures": 0,
        "relative_excluded": len(absolute) - len(relative_errors),
        "MAE": fmt(
            mp.fsum(absolute) / len(absolute)
        ),
        "MRE": (
            fmt(mp.fsum(relative_errors) / len(relative_errors))
            if relative_errors else ""
        ),
        "P95AE": fmt(percentile95(absolute)),
        "MaxAE": fmt(absolute[-1]),
        "MeanSignedError": fmt(
            mp.fsum(signed_errors) / len(signed_errors)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--omit-sum",
        action="store_true",
        help="Skip sum-of-products; execute only product-of-sums",
    )

    args = parser.parse_args()

    mp.dps = MP_DECIMAL_DIGITS

    assert INITIAL_LEVEL == 1
    assert all(n > 0 and n % 2 == 0 for n in N_VALUES)
    assert all(REPETITIONS[n] > 0 for n in N_VALUES)

    # By default, run both workloads.
    # With --omit-sum, run only product-of-sums.
    workloads = (
        ("product",)
        if args.omit_sum
        else ("sum", "product")
    )

    projects = [
        (workload, n, ROOT / f"{workload}_N{n}")
        for workload in workloads
        for n in N_VALUES
    ]

    for _, _, project in projects:
        if not (
            (project / "Nargo.toml").is_file()
            and (project / "src/main.nr").is_file()
        ):
            raise RuntimeError(
                f"Missing Nargo.toml or src/main.nr: {project}"
            )

    rows = []

    for workload, n, project in projects:
        repetitions = REPETITIONS[n]
        modes = (0,) if workload == "sum" else (0, 1)

        settings = [
            (d, r)
            for d in D_VALUES
            for r in modes
        ]

        signed_errors = {
            setting: [] for setting in settings
        }
        relative_errors = {
            setting: [] for setting in settings
        }

        print(
            f"{workload} N={n}: {repetitions} trials",
            flush=True,
        )

        for trial in range(repetitions):
            values, references = create_inputs(
                workload, n, trial
            )

            write_prover(project, values)
            actuals = execute(project, workload)

            for d, rounding in settings:
                reference = references[d]
                difference = actuals[d, rounding] - reference

                signed_errors[d, rounding].append(difference)

                if abs(reference) > mp.power(2, -d):
                    relative_errors[d, rounding].append(
                        abs(difference) / abs(reference)
                    )

            print(
                f"  trial {trial + 1}/{repetitions}",
                end="\r",
                flush=True,
            )

        print()

        for d, rounding in settings:
            rows.append(
                summarize(
                    workload,
                    n,
                    d,
                    rounding,
                    signed_errors[d, rounding],
                    relative_errors[d, rounding],
                )
            )

        if workload == "product":
            for d in D_VALUES:
                old = (
                    mp.fsum(abs(e) for e in signed_errors[d, 0])
                    / repetitions
                )
                new = (
                    mp.fsum(abs(e) for e in signed_errors[d, 1])
                    / repetitions
                )
                gain = (
                    (old - new) / old * 100
                    if old else mp.mpf(0)
                )

                print(
                    f"  D={d}: rounding MAE improvement = "
                    f"{mp.nstr(gain, 5)}%"
                )

    path = ROOT / "summary.csv"

    with path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Complete: {len(rows)} configuration summaries "
        f"saved to {path}"
    )


if __name__ == "__main__":
    main()
