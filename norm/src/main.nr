
use ssfp;
use ssfp::{SSFP_METADATA, SSFP};
use ssfp::SSFP_MATRIX;
use ssfp::Array_of_SSFP;

global N: u32 = 10000;

fn main(data: [Field; N]) -> pub [Field; N] {
    //min_max(data)
    zscore(data)
}


// #[allow(dead_code)]
// fn min_max(data: [Field; N]) -> [Field; N]{
//     let res = comptime{
//         ssfp::SSFP_CTX_INIT();
//         ssfp::SSFP_CTX_UPDATE(ssfp::SSFP_CTX_CONFIG{
//             CTX_PRIORITY_MODE: 0,
//             CTX_ZERO_AWARE_COMP: false,
//             CTX_RESET_ROUND: false,
//             CTX_INV_ROUND: false,
//             CTX_INV_PRECISION_LVL: 2,
//             CTX_LAZY_ZERO_ADJUST: false,
//             CTX_LN_LAZY_BOUNDS: false,
//         });

//         let d_md: SSFP_METADATA = SSFP_METADATA::unknown();
//         let mut d_arr: [SSFP; N] = [SSFP::default(); N];
//         for i in 0..N{
//             d_arr[i] = SSFP::init(1, d_md, f"data[{i}]");
//             d_arr[i].check();
//         }
//         let d_min: SSFP = d_arr.min();
//         let d_max: SSFP = d_arr.max();
//         let inv_denom: SSFP = (d_max - d_min).inv();

//         let mut norm_arr: [SSFP; N] = [SSFP::default(); N];
//         for i in 0..N{
//             norm_arr[i] = (d_arr[i] - d_min) * inv_denom;
//         }
//         ssfp::SSFP_PUSH_OUTPUTS(norm_arr);
//         ssfp::SSFP_CTX_SOLVE(false)
//     };
//     res
// }

// #[allow(dead_code)]
// fn zscore(data: [Field; N]) -> [Field; N]{
//     let res = comptime{
//         ssfp::SSFP_CTX_INIT();
//         ssfp::SSFP_CTX_UPDATE(ssfp::SSFP_CTX_CONFIG{
//             CTX_PRIORITY_MODE: 0,
//             CTX_ZERO_AWARE_COMP: false,
//             CTX_RESET_ROUND: false,
//             CTX_INV_ROUND: false,
//             CTX_INV_PRECISION_LVL: 2,
//             CTX_LAZY_ZERO_ADJUST: false,
//             CTX_LN_LAZY_BOUNDS: false,
//         });

//         let invN: SSFP = SSFP::new_from_int_const(1, N as Field).inv();

//         let d_md: SSFP_METADATA = SSFP_METADATA::unknown();
//         let mut d_arr: [SSFP; N] = [SSFP::default(); N];
//         for i in 0..N{
//             d_arr[i] = SSFP::init(1, d_md, f"data[{i}]");
//             d_arr[i].check();
//         }

//         // reseting here is more optimal than adjusting all levels of d_arr when subtracting
//         let avg: SSFP = (d_arr.reduce(|a,b| a + b) * invN).reset_full();
//         let mut var: SSFP = SSFP::default();
//         for i in 0..N{
//             let dif_avg: SSFP = d_arr[i] - avg;
//             var += (dif_avg * dif_avg);
//         }
//         var = (var * invN).sqrt();
//         let inv_var: SSFP = var.inv();

//         let mut norm_arr: [SSFP; N] = [SSFP::default(); N];
//         for i in 0..N{
//             norm_arr[i] = (d_arr[i] - avg) * inv_var;
//         }
//         ssfp::SSFP_PUSH_OUTPUTS(norm_arr);
//         ssfp::SSFP_CTX_SOLVE(false)
//     };
//     res
// }




// //-----------------------------------------------------
// // ALTERNATIVE MATRIX IMPLEMENTATIONS:

// #[allow(dead_code)]
// fn min_max(data: [Field; N]) -> [Field; N]{
//     let res = comptime{
//         ssfp::SSFP_CTX_INIT();
//         ssfp::SSFP_CTX_UPDATE(ssfp::SSFP_CTX_CONFIG{
//             CTX_PRIORITY_MODE: 0,
//             CTX_ZERO_AWARE_COMP: false,
//             CTX_RESET_ROUND: false,
//             CTX_INV_ROUND: false,
//             CTX_INV_PRECISION_LVL: 2,
//             CTX_LAZY_ZERO_ADJUST: false,
//             CTX_LN_LAZY_BOUNDS: false,
//         });

//         let d_md: SSFP_METADATA = SSFP_METADATA::unknown();
//         let mat: SSFP_MATRIX = SSFP_MATRIX::new_from_arr(1, N, 1, d_md, "data");
//         mat.check();
        
//         let mat_min: SSFP = mat.min();
//         let mat_max: SSFP = mat.max();
//         let inv_denom: SSFP = (mat_max - mat_min).inv();

//         let norm_mat: SSFP_MATRIX = mat.add_val(-mat_min).mult_val(inv_denom);

//         ssfp::SSFP_PUSH_OUTPUT(norm_mat.get_row(0));
//         ssfp::SSFP_CTX_SOLVE(false)
//     };
//     res
// }

#[allow(dead_code)]
fn zscore(data: [Field; N]) -> [Field; N]{
    let res = comptime{
        ssfp::SSFP_CTX_INIT();
        ssfp::SSFP_CTX_UPDATE(ssfp::SSFP_CTX_CONFIG{
            CTX_PRIORITY_MODE: 0,
            CTX_ZERO_AWARE_COMP: false,
            CTX_RESET_ROUND: false,
            CTX_INV_ROUND: false,
            CTX_INV_PRECISION_LVL: 2,
            CTX_LAZY_ZERO_ADJUST: false,
            CTX_LN_LAZY_BOUNDS: false,
        });

        let invN: SSFP = SSFP::new_from_int_const(1, N as Field).inv();

        let d_md: SSFP_METADATA = SSFP_METADATA::unknown();
        let mat: SSFP_MATRIX = SSFP_MATRIX::new_from_arr(1, N, 1, d_md, "data");
        mat.check();

        let avg: SSFP = mat.add_all() * invN;
        let avg_dif: SSFP_MATRIX = mat.add_val(-avg);
        let avg_dif_sq: SSFP_MATRIX = avg_dif.mult_elemwise(avg_dif);
        let var: SSFP = (avg_dif_sq.add_all() * invN).sqrt();
        let inv_var: SSFP = var.inv();

        let norm_mat: SSFP_MATRIX = avg_dif.mult_val(inv_var);

        ssfp::SSFP_PUSH_OUTPUT(norm_mat.get_row(0));
        ssfp::SSFP_CTX_SOLVE(false)
    };
    res
}
