
use ssfp;
use ssfp::{SSFP_METADATA, SSFP};
use ssfp::Array_of_SSFP;

global N: u32 = 10;

fn main(data: [Field; N]) -> pub [Field; N] {
    min_max(data)
}

fn min_max(data: [Field; N]) -> [Field; N]{
    let res = comptime{
        ssfp::SSFP_CTX_INIT();
        let d_md: SSFP_METADATA = SSFP_METADATA::unknown();
        let mut d_arr: [SSFP; N] = [SSFP::default(); N];
        for i in 0..N{
            d_arr[i] = SSFP::init(1, d_md, f"data[{i}]");
            d_arr[i].check();
        }
        let d_min: SSFP = d_arr.min();
        let d_max: SSFP = d_arr.max();
        let inv_denom: SSFP = (d_max - d_min).inv();

        let mut norm_arr: [SSFP; N] = [SSFP::default(); N];
        for i in 0..N{
            norm_arr[i] = (d_arr[i] - d_min) * inv_denom;
        }
        ssfp::SSFP_PUSH_OUTPUTS(norm_arr);
        ssfp::SSFP_CTX_SOLVE(false)
    };
    res
}
