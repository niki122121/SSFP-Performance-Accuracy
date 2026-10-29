#!/usr/bin/env python3
"""SSFP accuracy runner: 240 cases, two rounding modes, two references.

Requires a matching, already prepared src/main.nr with BATCH_SIZE inputs and
SSFP_RESULT <case_id> <sample_id> <field_value> <output_level> print lines.
The Noir contexts must follow configurations() exactly, including RR=IR=0/1.
Writes Prover.toml and runs nargo execute for every batch; outputs summary.csv.
"""

import csv
import hashlib
import math
import random
import re
import subprocess
from pathlib import Path

from mpmath import mp

# Experiment settings
SAMPLES_PER_CONFIGURATION = 10000
BATCH_SIZE = 100
RANDOM_SEED = 20261025
DECIMAL_DIGITS = 100
ORDINARY_FRACTION = 0.80
NARGO = "nargo"

ROOT = Path(__file__).resolve().parent
FIELD_MODULUS = 21888242871839275222246405745257275088548364400416034343698204186575808495617
RESULT_PATTERN = re.compile(r"SSFP_RESULT\s+(\d+)\s+(\d+)\s+(0[xX][0-9a-fA-F]+|\d+)\s+(\d+)\s*$")
OPS = ("inv", "div", "sqrt", "exp", "ln")
D_VALUES = (8, 16)
LEVELS = (1, 2, 3, 4)
ROUNDING = (0, 1)  # 0: RR=IR=0; 1: RR=IR=1


def configurations():
    """Must exactly match the case IDs in the static Noir main.nr."""
    cases = []
    for op in OPS:
        for d in D_VALUES:
            for level in LEVELS:
                if op in ("inv", "div"):
                    for precision in (1, 2, 3):
                        for mode in ("lazy", "strict"):
                            for rounding in ROUNDING:
                                cases.append((op, d, level, precision, mode, rounding))
                else:
                    for rounding in ROUNDING:
                        cases.append((op, d, level, None, None, rounding))
    return cases


def sample_inputs(op, d):
    """High-precision ordinary inputs + exact dyadic boundary inputs.

    Ordinary inputs are the same for D=8 and D=16; boundary inputs depend on D.
    The raw samples are identical across all levels and rounding configurations.
    """
    seed = int.from_bytes(hashlib.sha256(f"{RANDOM_SEED}|{op}".encode()).digest(), "big")
    rng = random.Random(seed)
    scale = 1 << d
    rows = []
    count_ordinary = round(SAMPLES_PER_CONFIGURATION * ORDINARY_FRACTION)

    def uniform(lo, hi):
        # 96 random fractional bits: no binary64 float input quantization.
        u = mp.mpf(rng.getrandbits(96)) / (1 << 96)
        return mp.mpf(lo) + (mp.mpf(hi) - mp.mpf(lo)) * u

    def signed_nonzero():
        # Stay nonzero even when quantized at the coarsest level (D=8, L=1).
        magnitude = uniform(mp.mpf(1) / 256, 8)
        return magnitude if rng.randrange(2) else -magnitude

    for _ in range(count_ordinary):
        if op == "inv":
            x, y = signed_nonzero(), None
        elif op == "div":
            x, y = uniform(-8, 8), signed_nonzero()
        elif op == "sqrt":
            x, y = uniform(0, 100), None
        elif op == "exp":
            x, y = uniform(-8, 8), None
        else:  # ln
            x, y = uniform(mp.mpf(1) / 256, 100), None
        rows.append((x, y, "ordinary"))

    # Keep the original script's reproducible, exactly representable boundaries.
    for i in range(SAMPLES_PER_CONFIGURATION - count_ordinary):
        if op in ("inv", "div"):
            candidates = [1, -1, 2, -2, 3, -3, 4, -4, scale - 1, scale,
                          scale + 1, -(scale - 1), -scale, -(scale + 1),
                          scale // 2, -(scale // 2), 2 * scale - 1, 2 * scale + 1]
            den = candidates[i % len(candidates)]
            num = [0, 1, -1, scale // 2, -(scale // 2), scale, -scale, 3 * scale][
                (i // len(candidates)) % 8]
            x, y = ((mp.mpf(den) / scale), None) if op == "inv" else (
                mp.mpf(num) / scale, mp.mpf(den) / scale)
        elif op == "sqrt":
            k = (i // 3) % 11
            x, y = mp.mpf(max(0, k * k * scale + (-1, 0, 1)[i % 3])) / scale, None
        elif op == "exp":
            k = (i // 3) % 43 - 21
            m = int(mp.nint(k * mp.log(2) * scale)) + (-1, 0, 1)[i % 3]
            x, y = mp.mpf(m) / scale, None
        else:  # ln
            candidates = [1, 2, 3, scale // 4, scale // 2, scale - 2,
                          scale - 1, scale, scale + 1, scale + 2,
                          2 * scale - 1, 2 * scale, 2 * scale + 1,
                          10 * scale, 100 * scale]
            x, y = mp.mpf(max(1, candidates[i % len(candidates)])) / scale, None
        rows.append((x, y, "boundary"))

    rng.shuffle(rows)
    return rows


def encoded_integer(x, d, level):
    # Truncate toward zero, matching the assumed SSFP input-encoding policy.
    return int(x * (1 << (d * level)))


def write_prover(cases, datasets, start):
    inputs = {}
    for op, d, level, _, _, _ in cases:
        name = f"x_{op}_D{d}_L{level}"
        if name not in inputs:
            inputs[name] = [encoded_integer(datasets[(op, d)][i][0], d, level) % FIELD_MODULUS
                            for i in range(start, start + BATCH_SIZE)]
        if op == "div":
            name = f"y_div_D{d}_L{level}"
            if name not in inputs:
                inputs[name] = [encoded_integer(datasets[(op, d)][i][1], d, level) % FIELD_MODULUS
                                for i in range(start, start + BATCH_SIZE)]
    (ROOT / "Prover.toml").write_text(
        "".join(name + " = [" + ", ".join('"0x{:x}"'.format(v) for v in values) + "]\n"
                for name, values in inputs.items()), encoding="utf-8")


def run_noir(case_count):
    result = subprocess.run([NARGO, "execute"], cwd=ROOT,
                            capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"nargo execute failed:\n{(result.stdout + result.stderr)[-4000:]}")

    outputs = {}
    for line in (result.stdout + "\n" + result.stderr).splitlines():
        if "SSFP_RESULT" not in line:
            continue
        match = RESULT_PATTERN.search(line.strip())
        if match is None:
            raise RuntimeError(f"Unexpected result format: {line}")
        case_id, sample_id, raw, level = (int(match[1]), int(match[2]),
                                          int(match[3], 0), int(match[4]))
        key = (case_id, sample_id)
        if key in outputs or not (0 <= case_id < case_count) or not (0 <= sample_id < BATCH_SIZE):
            raise RuntimeError(f"Duplicate or unknown result ID: {key}")
        if not (0 <= raw < FIELD_MODULUS and 0 <= level <= 64):
            raise RuntimeError(f"Invalid field value or level for {key}")
        outputs[key] = (raw, level)
    if len(outputs) != case_count * BATCH_SIZE:
        raise RuntimeError(f"Expected {case_count * BATCH_SIZE} results, received {len(outputs)}. "
                           "Check main.nr case order and BATCH_SIZE.")
    return outputs


def expected_value(op, x, y):
    if op == "inv":
        return 1 / x
    if op == "div":
        return x / y
    if op == "sqrt":
        return mp.sqrt(x)
    if op == "exp":
        return mp.exp(x)
    return mp.log(x)


def decoded_output(raw, level, d):
    signed = raw if raw <= FIELD_MODULUS // 2 else raw - FIELD_MODULUS
    return mp.mpf(signed) / (1 << (d * level))


def batch_references(datasets, start):
    """Compute each reference once per input/level, shared by all modes."""
    references = {}
    for op in OPS:
        for d in D_VALUES:
            samples = datasets[(op, d)][start:start + BATCH_SIZE]
            raw_refs = [expected_value(op, x, y) for x, y, _ in samples]
            for level in LEVELS:
                scale = 1 << (d * level)
                encoded_refs = []
                for x, y, _ in samples:
                    xe = mp.mpf(encoded_integer(x, d, level)) / scale
                    ye = None if y is None else mp.mpf(encoded_integer(y, d, level)) / scale
                    encoded_refs.append(expected_value(op, xe, ye))
                references[(op, d, level)] = (raw_refs, encoded_refs)
    return references


def percentile_95(values):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * 0.95
    lo = int(pos)
    hi = math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def summarize(cases, datasets, errors):
    summary = []
    for case_id, (op, d, level, precision, mode, rounding) in enumerate(cases):
        case_name = f"{op}_D{d}_L{level}"
        if precision is not None:
            case_name += f"_P{precision}_{mode}"
        case_name += f"_R{rounding}"
        for reference in ("raw", "encoded"):
            for group in ("all", "ordinary", "boundary"):
                selected = ("ordinary", "boundary") if group == "all" else (group,)
                abs_err = [v for g in selected for v in errors[(case_id, reference, g)][0]]
                rel_err = [v for g in selected for v in errors[(case_id, reference, g)][1]]
                count = (SAMPLES_PER_CONFIGURATION if group == "all" else
                         sum(row[2] == group for row in datasets[(op, d)]))
                display = lambda v: "" if v is None else format(v, ".14g")
                summary.append({
                    "case_id": case_id, "case": case_name, "operation": op,
                    "D": d, "L": level, "P": precision or "", "mode": mode or "",
                    "RR": rounding, "IR": rounding, "reference": reference, "group": group,
                    "samples": count, "valid": len(abs_err), "failures": count - len(abs_err),
                    "relative_excluded": len(abs_err) - len(rel_err), "relative_n": len(rel_err),
                    "MAE": display(math.fsum(abs_err) / len(abs_err)) if abs_err else "",
                    "MRE": display(math.fsum(rel_err) / len(rel_err)) if rel_err else "",
                    "P95AE": display(percentile_95(abs_err)) if abs_err else "",
                    "MaxAE": display(max(abs_err)) if abs_err else "",
                })
    return summary


def main():
    mp.dps = DECIMAL_DIGITS
    assert 0 < BATCH_SIZE <= SAMPLES_PER_CONFIGURATION
    assert SAMPLES_PER_CONFIGURATION % BATCH_SIZE == 0
    if not (ROOT / "Nargo.toml").is_file() or not (ROOT / "src/main.nr").is_file():
        raise SystemExit("Place this script in your Noir project beside Nargo.toml")

    cases = configurations()
    datasets = {(op, d): sample_inputs(op, d) for op in OPS for d in D_VALUES}
    errors = {(i, ref, group): ([], []) for i in range(len(cases))
              for ref in ("raw", "encoded") for group in ("ordinary", "boundary")}
    threshold = {d: mp.power(2, -d) for d in D_VALUES}

    print(f"{len(cases)} cases, {SAMPLES_PER_CONFIGURATION} samples/case, "
          f"{SAMPLES_PER_CONFIGURATION // BATCH_SIZE} executions; RR=IR in {{0, 1}}")
    for batch in range(SAMPLES_PER_CONFIGURATION // BATCH_SIZE):
        start = batch * BATCH_SIZE
        print(f"Batch {batch + 1}/{SAMPLES_PER_CONFIGURATION // BATCH_SIZE}", flush=True)
        write_prover(cases, datasets, start)
        outputs = run_noir(len(cases))
        refs = batch_references(datasets, start)

        for case_id, (op, d, level, _, _, _) in enumerate(cases):
            raw_refs, encoded_refs = refs[(op, d, level)]
            for sample_id in range(BATCH_SIZE):
                group = datasets[(op, d)][start + sample_id][2]
                raw, output_level = outputs[(case_id, sample_id)]
                actual = decoded_output(raw, output_level, d)
                for reference_name, reference in (("raw", raw_refs[sample_id]),
                                                  ("encoded", encoded_refs[sample_id])):
                    ae, re = errors[(case_id, reference_name, group)]
                    absolute = abs(actual - reference)
                    ae.append(float(absolute))
                    if abs(reference) > threshold[d]:
                        re.append(float(absolute / abs(reference)))

    summary = summarize(cases, datasets, errors)
    path = ROOT / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    print(f"Finished: {len(cases)} cases x {SAMPLES_PER_CONFIGURATION} samples")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
