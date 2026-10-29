// AUTO-GENERATED: sum, N=100; no explicit resets.
use ssfp;
use ssfp::{SSFP, SSFP_METADATA, SSFP_CTX_CONFIG, SSFP_CTX_INIT_CUSTOM, SSFP_PUSH_OUTPUTS, SSFP_CTX_SOLVE};
use std::meta::ctstring::AsCtString;

fn main(a8: [Field; 100], b8: [Field; 100], a16: [Field; 100], b16: [Field; 100]) {

    // D=8, R0; automatic SSFP precision management.
    let result_8_r0 = comptime {
        let conf = SSFP_CTX_CONFIG {
            CTX_PRIORITY_MODE: 0,
            CTX_ZERO_AWARE_COMP: false,
            CTX_RESET_ROUND: false,
            CTX_INV_ROUND: false,
            CTX_INV_PRECISION_LVL: 2,
            CTX_LAZY_ZERO_ADJUST: false,
            CTX_LN_LAZY_BOUNDS: false,
        };
        SSFP_CTX_INIT_CUSTOM(32, 8, conf);

        let a0: SSFP = SSFP::init(1, SSFP_METADATA::unknown(), "a8[0]");
        let b0: SSFP = SSFP::init(1, SSFP_METADATA::unknown(), "b8[0]");
        let mut acc: SSFP = a0 * b0;

        for i in 1..100 {
            let a: SSFP = SSFP::init(1, SSFP_METADATA::unknown(), f"a8[{i}]".as_ctstring());
            let b: SSFP = SSFP::init(1, SSFP_METADATA::unknown(), f"b8[{i}]".as_ctstring());
            acc = acc + a * b;
        }

        SSFP_PUSH_OUTPUTS([acc]);
        SSFP_CTX_SOLVE(false)
    };

    let levels_8_r0: [u32] = comptime { ssfp::SSFP_GET_OUT_LVLS_FORCE() };
    let level_8_r0 = levels_8_r0[0];
    println(f"SSFP_PROP 8 0 {result_8_r0} {level_8_r0}");

    // D=16, R0; automatic SSFP precision management.
    let result_16_r0 = comptime {
        let conf = SSFP_CTX_CONFIG {
            CTX_PRIORITY_MODE: 0,
            CTX_ZERO_AWARE_COMP: false,
            CTX_RESET_ROUND: false,
            CTX_INV_ROUND: false,
            CTX_INV_PRECISION_LVL: 2,
            CTX_LAZY_ZERO_ADJUST: false,
            CTX_LN_LAZY_BOUNDS: false,
        };
        SSFP_CTX_INIT_CUSTOM(32, 16, conf);

        let a0: SSFP = SSFP::init(1, SSFP_METADATA::unknown(), "a16[0]");
        let b0: SSFP = SSFP::init(1, SSFP_METADATA::unknown(), "b16[0]");
        let mut acc: SSFP = a0 * b0;

        for i in 1..100 {
            let a: SSFP = SSFP::init(1, SSFP_METADATA::unknown(), f"a16[{i}]".as_ctstring());
            let b: SSFP = SSFP::init(1, SSFP_METADATA::unknown(), f"b16[{i}]".as_ctstring());
            acc = acc + a * b;
        }

        SSFP_PUSH_OUTPUTS([acc]);
        SSFP_CTX_SOLVE(false)
    };

    let levels_16_r0: [u32] = comptime { ssfp::SSFP_GET_OUT_LVLS_FORCE() };
    let level_16_r0 = levels_16_r0[0];
    println(f"SSFP_PROP 16 0 {result_16_r0} {level_16_r0}");
}
