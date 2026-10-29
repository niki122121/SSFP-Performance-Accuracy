
#!/usr/bin/env python3
"""Generate complete Noir projects for natural SSFP error propagation."""

import argparse
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
N_VALUES = (10, 100, 1000, 10000)


def context(workload, n, d, rounding):
    tag = f"{d}_r{rounding}"
    first = "a0 * b0" if workload == "sum" else "a0 + b0"
    update = "acc + a * b" if workload == "sum" else "acc * (a + b)"
    flag = str(bool(rounding)).lower()

    return f"""
    // D={d}, R{rounding}; automatic SSFP precision management.
    let result_{tag} = comptime {{
        let conf = SSFP_CTX_CONFIG {{
            CTX_PRIORITY_MODE: 0,
            CTX_ZERO_AWARE_COMP: false,
            CTX_RESET_ROUND: {flag},
            CTX_INV_ROUND: {flag},
            CTX_INV_PRECISION_LVL: 2,
            CTX_LAZY_ZERO_ADJUST: false,
            CTX_LN_LAZY_BOUNDS: false,
        }};
        SSFP_CTX_INIT_CUSTOM(32, {d}, conf);

        let a0: SSFP = SSFP::init(1, SSFP_METADATA::unknown(), "a{d}[0]");
        let b0: SSFP = SSFP::init(1, SSFP_METADATA::unknown(), "b{d}[0]");
        let mut acc: SSFP = {first};

        for i in 1..{n} {{
            let a: SSFP = SSFP::init(1, SSFP_METADATA::unknown(), f"a{d}[{{i}}]".as_ctstring());
            let b: SSFP = SSFP::init(1, SSFP_METADATA::unknown(), f"b{d}[{{i}}]".as_ctstring());
            acc = {update};
        }}

        SSFP_PUSH_OUTPUTS([acc]);
        SSFP_CTX_SOLVE(false)
    }};

    let levels_{tag}: [u32] = comptime {{ ssfp::SSFP_GET_OUT_LVLS_FORCE() }};
    let level_{tag} = levels_{tag}[0];
    println(f"SSFP_PROP {d} {rounding} {{result_{tag}}} {{level_{tag}}}");
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ssfp-path",
        type=Path,
        default=ROOT / "../../Noir-SSFP",
        help="Path to the SSFP library (default: ./ssfp)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Also compile all generated Noir projects",
    )
    args = parser.parse_args()

    library = args.ssfp_path.resolve()
    if not (library / "Nargo.toml").is_file():
        parser.error(
            f"SSFP library not found at {library}. "
            "Specify --ssfp-path PATH_TO_SSFP_LIBRARY"
        )

    for workload in ("sum", "product"):
        for n in N_VALUES:
            project = ROOT / f"{workload}_N{n}"
            (project / "src").mkdir(parents=True, exist_ok=True)

            modes = (0,) if workload == "sum" else (0, 1)

            intro = f"""// AUTO-GENERATED: {workload}, N={n}; no explicit resets.
use ssfp;
use ssfp::{{SSFP, SSFP_METADATA, SSFP_CTX_CONFIG, SSFP_CTX_INIT_CUSTOM, SSFP_PUSH_OUTPUTS, SSFP_CTX_SOLVE}};
use std::meta::ctstring::AsCtString;

fn main(a8: [Field; {n}], b8: [Field; {n}], a16: [Field; {n}], b16: [Field; {n}]) {{
"""

            program = intro + "".join(
                context(workload, n, d, r)
                for d in (8, 16)
                for r in modes
            ) + "}\n"

            (project / "src/main.nr").write_text(
                program, encoding="utf-8"
            )

            # Compute the correct relative dependency path.
            rel_library = Path(
                os.path.relpath(library, project)
            ).as_posix()

            manifest = (
                f'[package]\n'
                f'name = "ssfp_{workload}_n{n}"\n'
                f'type = "bin"\n'
                f'authors = []\n\n'
                f'[dependencies]\n'
                f'ssfp = {{ path = "{rel_library}" }}\n'
            )

            (project / "Nargo.toml").write_text(
                manifest, encoding="utf-8"
            )

            print(
                f"Generated {project.name}: {2 * len(modes)} contexts",
                flush=True,
            )

            if args.compile:
                print(f"  Compiling {project.name} ...", flush=True)
                subprocess.run(
                    ["nargo", "compile"],
                    cwd=project,
                    check=True,
                )

    print("Done. Run python run_propagation.py after compilation.")


if __name__ == "__main__":
    main()
