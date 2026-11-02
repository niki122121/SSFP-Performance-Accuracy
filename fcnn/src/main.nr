use ssfp::{SSFP, SSFP_METADATA, SSFP_MATRIX};
use ssfp::{WHOLE_BITS, DECIMAL_BITS, INV_PRECISION_LVL, lvl_bits};
use ssfp::{SSFP_CTX_INIT, SSFP_CTX_SOLVE, SSFP_PUSH_OUTPUT};
use ssfp::ssfp_context;
use std::{meta::ctstring::AsCtString, static_assert};

comptime global TEST_MODE: bool = true; //no hashing for faster testing
comptime global TEST_PRINT_ALL: bool = true; //when testing we can print weights, biases, loss... (this dumps a lot of text)

comptime global INPUT_SIZE: u32 = 784;
comptime global HIDDEN_SIZE_1: u32 = 64;
comptime global HIDDEN_SIZE_2: u32 = 32;
comptime global OUTPUT_SIZE: u32 = 10;
comptime global OUTPUT_SIZE_BITS: u32 = 4;
comptime global BATCH_SIZE: u32 = 16;
comptime global DZ1_BITS: u128 = 26;
comptime global MAX_TRAINING_BATCHES: u32 = 9380;

global OUTPUT_SIZE_RT: u32 = comptime{OUTPUT_SIZE};
global OUTPUT_SIZE_BITS_RT: u32 = comptime{OUTPUT_SIZE_BITS};
global BATCH_SIZE_RT: u32 = comptime{BATCH_SIZE};

comptime mut global INV_256: SSFP = SSFP::default();
comptime mut global LEARNING_RATE: SSFP = SSFP::default();
comptime mut global BATCH_SIZE_INV: SSFP = SSFP::default();

comptime global SSFP_MD_DEFAULT: SSFP_METADATA = SSFP_METADATA::default();
comptime global SSFP_MD_NEG: SSFP_METADATA = SSFP_METADATA{sgnKnown: true, sgnPos: false};
comptime global SSFP_MD_UNKNOWN: SSFP_METADATA = SSFP_METADATA{sgnKnown: false, sgnPos: true};

global TOTAL_LENG: u32 = INPUT_SIZE * HIDDEN_SIZE_1 + 
HIDDEN_SIZE_1 * HIDDEN_SIZE_2 + HIDDEN_SIZE_2 * OUTPUT_SIZE + 
HIDDEN_SIZE_1 + HIDDEN_SIZE_2 + OUTPUT_SIZE;

comptime fn pack_per_field(lvl: u32) -> u32 {
    let BITS: u32 = lvl_bits(lvl);
    let DIGIT_BITS: u32 = BITS + 1;
    let FIELD_BITS: u32 = (std::field::modulus_num_bits() as u32) - 1;
    static_assert(DIGIT_BITS <= FIELD_BITS, "SSFP value is too wide for packing");
    FIELD_BITS / DIGIT_BITS
}
comptime fn packed_len(M: u32, N: u32, lvl: u32) -> u32 {
    let K: u32 = pack_per_field(lvl);
    (M * N + K - 1) / K
}
global W1_LVL: u32 = 2;
global OTHER_LVL: u32 = 1;
global PACKED_TOTAL: u32 =
    packed_len(INPUT_SIZE, HIDDEN_SIZE_1, W1_LVL) +
    packed_len(HIDDEN_SIZE_1, HIDDEN_SIZE_2, OTHER_LVL) +
    packed_len(HIDDEN_SIZE_2, OUTPUT_SIZE, OTHER_LVL) +
    packed_len(1, HIDDEN_SIZE_1, OTHER_LVL) +
    packed_len(1, HIDDEN_SIZE_2, OTHER_LVL) +
    packed_len(1, OUTPUT_SIZE, OTHER_LVL);

global K_W1: u32 = pack_per_field(W1_LVL);
global K_OTHERS: u32 = pack_per_field(OTHER_LVL);

mod tests;

// ----------------------------------------------------------------------------------

comptime fn relu(mat_ssfp: SSFP_MATRIX) -> SSFP_MATRIX{
    let (M, N) = (mat_ssfp.M, mat_ssfp.N);
    if mat_ssfp.md.sgnKnown{
        if mat_ssfp.md.sgnPos{
            mat_ssfp
        }else{
            SSFP_MATRIX::new_default(M, N, 0, 0)
        }
    }else{
        let (mat, BITS): (CtString, u32) = (mat_ssfp.mat, lvl_bits(mat_ssfp.lvl));
        let res_ij: CtString = f"ssfp::relu_sgnU::<{BITS}>({mat}[i][j])";
        SSFP_MATRIX::new_from_double_loop(M, N, mat_ssfp.lvl, SSFP_MD_DEFAULT, res_ij)
    }
}
comptime fn relu_der(mat_ssfp: SSFP_MATRIX) -> SSFP_MATRIX{
    let (M, N) = (mat_ssfp.M, mat_ssfp.N);
    if mat_ssfp.md.sgnKnown{
        if mat_ssfp.md.sgnPos{
            SSFP_MATRIX::new_default(M, N, 1, 0)
        }else{
            SSFP_MATRIX::new_default(M, N, 0, 0)
        }
    }else{
        let (mat, BITS): (CtString, u32) = (mat_ssfp.mat, lvl_bits(mat_ssfp.lvl));
        let res_ij: CtString = f"ssfp::relu_der_sgnU::<{BITS}>({mat}[i][j])";
        SSFP_MATRIX::new_from_double_loop(M, N, 0, SSFP_MD_DEFAULT, res_ij)
    }
}

#[allow(dead_code)]
fn sub_max_by_rows<let N_SAMPLES: u32, let BITS: u32>
    (logits: [[Field; OUTPUT_SIZE_RT]; N_SAMPLES]) -> [[Field; OUTPUT_SIZE_RT]; N_SAMPLES]
{
    let mut logits_adj: [[Field; OUTPUT_SIZE_RT]; N_SAMPLES] = [[0; OUTPUT_SIZE_RT]; N_SAMPLES];
    for i in 0..N_SAMPLES{
        let v_arr: [Field; OUTPUT_SIZE_RT] = logits[i];
        let maxVal: Field = ssfp::ssfp_internal::max_unchecked(v_arr);
        for j in 0..OUTPUT_SIZE_RT{
            (maxVal - v_arr[j]).assert_max_bit_size::<BITS + 1>();
            logits_adj[i][j] = logits[i][j] - maxVal;
        }
    }
    logits_adj
}
comptime fn softmax<let N_SAMPLES: u32>(logits: SSFP_MATRIX) -> SSFP_MATRIX{
    let (l_mat, BITS): (CtString, u32) = (logits.mat, lvl_bits(logits.lvl));
    let logits_adj_op: CtString = f"sub_max_by_rows::<{N_SAMPLES},{BITS}>({l_mat})";
    let logits_adj: SSFP_MATRIX = SSFP_MATRIX::new(logits.M, logits.N, logits.lvl, SSFP_MD_NEG, logits_adj_op);
    let exponentials: SSFP_MATRIX = logits_adj.exp();
    let exp_sum: SSFP_MATRIX = exponentials.add_cols();
    let exp_sum_inv: SSFP_MATRIX = exp_sum.inv();

    let (exp_mat, exp_sum_inv_mat): (CtString, CtString) = (exponentials.mat, exp_sum_inv.mat);
    let lvl_res: u32 = exponentials.lvl + INV_PRECISION_LVL;
    let res_ij: CtString = f"{exp_mat}[i][j] * {exp_sum_inv_mat}[i][0]"; // compute probabilities
    SSFP_MATRIX::new_from_double_loop(N_SAMPLES, OUTPUT_SIZE, lvl_res, SSFP_MD_DEFAULT, res_ij)
}


#[allow(dead_code)]
fn select_correct_probs_rt(
    probabilites: [[Field; OUTPUT_SIZE_RT]; BATCH_SIZE_RT], labels_adj: [[Field; OUTPUT_SIZE_RT]; BATCH_SIZE_RT]
) -> [Field; BATCH_SIZE_RT]{
    let mut correct_probs: [Field; BATCH_SIZE_RT] = [0; BATCH_SIZE_RT];
    for i in 0..BATCH_SIZE_RT{
        for j in 0..OUTPUT_SIZE_RT{
            correct_probs[i] = correct_probs[i] + probabilites[i][j] * labels_adj[i][j];
        }
    }
    correct_probs
}
comptime fn cross_entropy_loss(probabilites: SSFP_MATRIX, labels_adj: SSFP_MATRIX) -> SSFP{
    let probabilites_mat: CtString = probabilites.mat;
    let labels_adj_mat: CtString = labels_adj.mat;
    let correct_probs_res: CtString = f"select_correct_probs_rt({probabilites_mat},{labels_adj_mat})";
    let correct_probs_var: CtString = ssfp_context::append_ctx_Var(f"[Field; {BATCH_SIZE}]", correct_probs_res);
    let correct_probs: SSFP_MATRIX = SSFP_MATRIX::new_from_arr(1, BATCH_SIZE, probabilites.lvl, probabilites.md, correct_probs_var);

    let loss: SSFP = -(correct_probs.ln().add_cols().get_val(0,0) * BATCH_SIZE_INV);
    loss
}


comptime fn forward<let N_SAMPLES: u32>(
    x: SSFP_MATRIX, W1: SSFP_MATRIX, W2: SSFP_MATRIX, W3: SSFP_MATRIX, b1: SSFP_MATRIX, b2: SSFP_MATRIX, b3: SSFP_MATRIX)
 -> (SSFP_MATRIX, SSFP_MATRIX, SSFP_MATRIX, SSFP_MATRIX, SSFP_MATRIX){
    let z1: SSFP_MATRIX = (x * W1).add_broadcasted(b1);
    let a1: SSFP_MATRIX = relu(z1);

    let z2: SSFP_MATRIX = (a1 * W2).add_broadcasted(b2);
    let a2: SSFP_MATRIX = relu(z2);

    let z3: SSFP_MATRIX = (a2 * W3).add_broadcasted(b3);
    let probabilites: SSFP_MATRIX = softmax::<N_SAMPLES>(z3);

    (probabilites, z1, a1, z2, a2)
}


comptime fn backward(
    x: SSFP_MATRIX, labels_adj: SSFP_MATRIX, probabilites: SSFP_MATRIX, 
    z1: SSFP_MATRIX, a1: SSFP_MATRIX, z2: SSFP_MATRIX, a2: SSFP_MATRIX,
    W2: SSFP_MATRIX, W3: SSFP_MATRIX
) -> (SSFP_MATRIX, SSFP_MATRIX, SSFP_MATRIX, SSFP_MATRIX, SSFP_MATRIX, SSFP_MATRIX){
    // Note the multiplication by Learning Rate here. It propagates downwards to all our matrices and we avoid
    // having to do it at the end (tipically W -= dW * LR, but here it is already backed in).
    // Also, we do a manual reset (by testing it gave us slightly less acir opcodes)
    let dz3: SSFP_MATRIX = (probabilites - labels_adj).mult_val(BATCH_SIZE_INV).mult_val(LEARNING_RATE).reset_full();
    let dW3: SSFP_MATRIX = a2.mult(dz3, true, false);
    let db3: SSFP_MATRIX = dz3.add_rows();
    let da2: SSFP_MATRIX = dz3.mult(W3, false, true);

    let dz2: SSFP_MATRIX = da2.mult_elemwise(relu_der(z2));
    let dW2: SSFP_MATRIX = a1.mult(dz2, true, false);
    let db2: SSFP_MATRIX = dz2.add_rows();
    let da1: SSFP_MATRIX = dz2.mult(W2, false, true);

    // manual reset of dz1 is purely for convinience as it makes level dW1 = 2
    // This coincides with the starting level of W1 and allows us to avoid an expensive reset of all it's elements
    let dz1: SSFP_MATRIX = da1.mult_elemwise(relu_der(z1)).reset_full();
    // This check is key as it constarins the updated weights 1 without explicitly calling .check() on the entire weight matrix
    // the chosen DZ1_BITS combined with the know ranges of x and the know number of batches to be performed when training
    // allows us to establish this limit.
    dz1.check_raw_bit_size(DZ1_BITS as u32);
    let dW1: SSFP_MATRIX = x.mult(dz1, true, false);
    let db1: SSFP_MATRIX = dz1.add_rows();

    (dW1, db1, dW2, db2, dW3, db3)
}




unconstrained fn arr_colwise_max_rt<let N_SAMPLES: u32>(mat: [[Field; OUTPUT_SIZE_RT]; N_SAMPLES]) -> [Field; N_SAMPLES]{
    let mut indxs: [Field; N_SAMPLES] = [0; N_SAMPLES];
    for i in 0..N_SAMPLES{
        let mut val: Field = mat[i][0];
        for j in 0..OUTPUT_SIZE_RT{
            if ssfp::lt_custom_all_unc(val, mat[i][j]){
                val = mat[i][j];
                indxs[i] = j as Field;
            }
        }
    }
    indxs
}
#[allow(dead_code)]
fn accuracy_count_rt<let N_SAMPLES: u32, let BITS: u32>(probabilites: [[Field; OUTPUT_SIZE_RT]; N_SAMPLES], labels: [Field; N_SAMPLES]) -> Field{
    let mut correct_count: Field = 0;
    // Safety: max val indices are properly checked and constrained
    let indxs: [Field; N_SAMPLES] = unsafe{arr_colwise_max_rt::<N_SAMPLES>(probabilites)};
    for i in 0..N_SAMPLES{
        let maxVal_i: Field = probabilites[i][indxs[i] as u32];
        for j in 0..OUTPUT_SIZE_RT {
            assert(ssfp::sign_abs::<BITS,0,1>(maxVal_i - probabilites[i][j]).0 == 1);
        }
        correct_count = correct_count + ((indxs[i] == labels[i]) as Field);
    }
    correct_count
}
#[allow(dead_code)]
comptime fn accuracy_count<let N_SAMPLES: u32>(probabilites: SSFP_MATRIX, labels: CtString) -> CtString{
    let probabilites_mat: CtString = probabilites.mat;
    let probabilites_BITS: u32 = lvl_bits(probabilites.lvl);
    ssfp_context::append_ctx_var_Field(f"accuracy_count_rt::<{N_SAMPLES},{probabilites_BITS}>({probabilites_mat},{labels})")
}




//-----------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------


unconstrained fn adjust_labels_unc<let N_SAMPLES: u32>(labels: [Field; N_SAMPLES]) -> [[Field; OUTPUT_SIZE_RT]; N_SAMPLES]{
    let mut res: [[Field; OUTPUT_SIZE_RT]; N_SAMPLES] = [[0; OUTPUT_SIZE_RT]; N_SAMPLES];
    for i in 0..N_SAMPLES{
        res[i][labels[i] as u32] = 1;
    }
    res
}
#[allow(dead_code)]
fn validate_and_adjust_labels_rt<let N_SAMPLES: u32>(labels: [Field; N_SAMPLES]) -> [[Field; OUTPUT_SIZE_RT]; N_SAMPLES]{
    let MAX_LABEL: Field = (OUTPUT_SIZE_RT as Field) - 1Field;
    // Safety: res is properly checked and contrained later when invoked
    let labels_adj: [[Field; OUTPUT_SIZE_RT]; N_SAMPLES] = unsafe{adjust_labels_unc::<N_SAMPLES>(labels)};

    for i in 0..N_SAMPLES{
        // make sure label values are in range [0, MAX_LABEL];
        labels[i].assert_max_bit_size::<OUTPUT_SIZE_BITS_RT>();
        (MAX_LABEL - labels[i]).assert_max_bit_size::<OUTPUT_SIZE_BITS_RT>();

        // constrain bits from unsafe function and check that they are all zeroes except one (which is equal to 1)
        for j in 0..OUTPUT_SIZE_RT{
            labels_adj[i][j].assert_max_bit_size::<1>();
        }
        assert_eq(labels_adj[i].reduce(|a, b| a + b), 1);

        // make sure that the position of the 1 matches the correct label value
        for j in 0..OUTPUT_SIZE_RT{
            assert_eq(labels[i] * labels_adj[i][j], (j as Field) * labels_adj[i][j]);
        }
    }
    labels_adj
}
comptime fn validate_and_adjust_labels<let N_SAMPLES: u32>(labels_train_ctstr: CtString) -> SSFP_MATRIX{
    let l_adj_ctstr: CtString = f"validate_and_adjust_labels_rt::<{N_SAMPLES}>({labels_train_ctstr})";
    let l_adj_ctstr_var: CtString = ssfp_context::append_ctx_Var(f"[[Field; {OUTPUT_SIZE}]; {N_SAMPLES}]", l_adj_ctstr);
    SSFP_MATRIX::new(N_SAMPLES, OUTPUT_SIZE, 0, SSFP_MD_DEFAULT, l_adj_ctstr_var)
}


pub fn train_batch(
    data_train: [[Field; INPUT_SIZE]; BATCH_SIZE],
    labels_train: [Field; BATCH_SIZE],
    W1_raw: [[Field; HIDDEN_SIZE_1]; INPUT_SIZE],
    W2_raw: [[Field; HIDDEN_SIZE_2]; HIDDEN_SIZE_1],
    W3_raw: [[Field; OUTPUT_SIZE]; HIDDEN_SIZE_2],
    b1_raw: [Field; HIDDEN_SIZE_1],
    b2_raw: [Field; HIDDEN_SIZE_2],
    b3_raw: [Field; OUTPUT_SIZE]
) -> [Field; 3]{
    let res = comptime{
        static_assert(
            (BATCH_SIZE as u128) * (MAX_TRAINING_BATCHES as u128)* (1u128 << DZ1_BITS)
            <= (1u128 << ((WHOLE_BITS + DECIMAL_BITS - 2) as u128)),
            "DZ1_BITS is too large to guarantee W1 remains in range"
        );

        SSFP_CTX_INIT();

        INV_256 = SSFP::new_from_int_const(1, 255).inv();
        LEARNING_RATE = SSFP::new_from_int_const(1, 50).inv(); // 0.02
        BATCH_SIZE_INV = SSFP::new_from_int_const(1, BATCH_SIZE as Field).inv();

        // W1 is level 2 so we avoid an expensive reset at the en of the batch when updating and hashing weights
        // Note we do NOT check the weight and bias matrices as the check is essentially done by the hash.
        // Otherwise we would need to invoke .check()
        let W1: SSFP_MATRIX = SSFP_MATRIX::init(INPUT_SIZE, HIDDEN_SIZE_1, 2, SSFP_MD_DEFAULT, "W1_raw");
        let W2: SSFP_MATRIX = SSFP_MATRIX::init(HIDDEN_SIZE_1, HIDDEN_SIZE_2, 1, SSFP_MD_DEFAULT, "W2_raw");
        let W3: SSFP_MATRIX = SSFP_MATRIX::init(HIDDEN_SIZE_2, OUTPUT_SIZE, 1, SSFP_MD_DEFAULT, "W3_raw");

        let b1: SSFP_MATRIX = SSFP_MATRIX::new_from_arr(1, HIDDEN_SIZE_1, 1, SSFP_MD_UNKNOWN, "b1_raw");
        let b2: SSFP_MATRIX = SSFP_MATRIX::new_from_arr(1, HIDDEN_SIZE_2, 1, SSFP_MD_UNKNOWN, "b2_raw");
        let b3: SSFP_MATRIX = SSFP_MATRIX::new_from_arr(1, OUTPUT_SIZE, 1, SSFP_MD_UNKNOWN, "b3_raw");


        
        let x_train_raw: SSFP_MATRIX = SSFP_MATRIX::init_from_ints(BATCH_SIZE, INPUT_SIZE, SSFP_MD_DEFAULT, "data_train");
        x_train_raw.check_raw_bit_size(8);
        let x_train: SSFP_MATRIX = x_train_raw.mult_val(INV_256);

        let labels_adj_train: SSFP_MATRIX = validate_and_adjust_labels::<BATCH_SIZE>("labels_train");

        // BATCH TRAIN ---------------------------------------------------------------------------------------------------------------------------
        let (probabilities, z1, a1, z2, a2) = forward::<BATCH_SIZE>(x_train, W1, W2, W3, b1, b2, b3);

        let loss: SSFP = cross_entropy_loss(probabilities, labels_adj_train);

        let (dW1, db1, dW2, db2, dW3, db3) = 
        backward(x_train, labels_adj_train, probabilities, z1, a1, z2, a2, W2, W3);

        // Update 
        // Note we dont adjust by Learning Rate as this is done in backpropagation automatically
        // Also note that W1 is not reset. We try to avoid reseting this larger matrix and decide to
        // set W1 as well as dW1 to lvl 2
        let W1_updated: SSFP_MATRIX = W1 - dW1;
        let b1_updated: SSFP_MATRIX = (b1 - db1).reset_full();
        let W2_updated: SSFP_MATRIX = (W2 - dW2).reset_full();
        let b2_updated: SSFP_MATRIX = (b2 - db2).reset_full();
        let W3_updated: SSFP_MATRIX = (W3 - dW3).reset_full();
        let b3_updated: SSFP_MATRIX = (b3 - db3).reset_full();

        let lvl_err_msg: CtString = "Levels must match so compiled code can be reused for more batches and epochs";
        static_assert(W1.lvl == W1_updated.lvl, lvl_err_msg);
        static_assert(b1.lvl == b1_updated.lvl, lvl_err_msg);
        static_assert(W2.lvl == W2_updated.lvl, lvl_err_msg);
        static_assert(b2.lvl == b2_updated.lvl, lvl_err_msg);
        static_assert(W3.lvl == W3_updated.lvl, lvl_err_msg);
        static_assert(b3.lvl == b3_updated.lvl, lvl_err_msg);


        SSFP_PUSH_OUTPUT(loss);
        if TEST_MODE{
            // no need for hashing in test mode
            SSFP_PUSH_OUTPUT("0".as_ctstring());
            SSFP_PUSH_OUTPUT("0".as_ctstring());
            if TEST_PRINT_ALL{
                // we can see more params and inspect manually if needed
                ssfp_context::append_ctx_println(loss.v);
                ssfp_context::append_ctx_println(probabilities.mat);
                ssfp_context::append_ctx_println(W1_updated.mat);
                ssfp_context::append_ctx_println(b1_updated.mat);
                ssfp_context::append_ctx_println(W2_updated.mat);
                ssfp_context::append_ctx_println(b2_updated.mat);
                ssfp_context::append_ctx_println(W3_updated.mat);
                ssfp_context::append_ctx_println(b3_updated.mat);
            }
        }else{
            SSFP_PUSH_OUTPUT(hash_all(W1, W2, W3, b1, b2, b3));
            SSFP_PUSH_OUTPUT(hash_all(W1_updated, W2_updated, W3_updated, b1_updated, b2_updated, b3_updated));
        }
        SSFP_CTX_SOLVE(false)
    };
    res
}



// Hashing code -----------------------------------------------------
#[allow(dead_code)]
fn flatten_raw<let M: u32, let N: u32, let IDX: u32>(flat_arr: &mut [Field; TOTAL_LENG], mat: [[Field; N]; M]){
    for i in 0..M{
        for j in 0..N{
            flat_arr[i + M*j + IDX] = mat[i][j];
        }
    }
}
#[allow(dead_code)]
fn pack<let M: u32, let N: u32, let START: u32, let BITS: u32, let K: u32>
    (packed_arr: &mut [Field; PACKED_TOTAL], mat: [[Field; N]; M]) 
{
    let FULL_GROUPS: u32 = M * N / K;
    let REM: u32 = M * N % K;

    let OFFSET: Field = 2.pow_32(BITS as Field);
    let BASE: Field = OFFSET * 2;

    // Pack all complete groups.
    for g in 0..FULL_GROUPS {
        let mut acc: Field = 0;

        for s in 0..K {
            let linear: u32 = g * K + K - 1 - s;
            let i: u32 = linear % M;
            let j: u32 = linear / M;

            acc = acc * BASE + mat[i][j] + OFFSET;
        }

        packed_arr[START + g] = acc;
    }

    // Pack the remaining elements, if any
    if REM != 0 {
        let mut acc: Field = 0;

        for s in 0..REM {
            let linear: u32 = FULL_GROUPS * K + REM - 1 - s;
            let i: u32 = linear % M;
            let j: u32 = linear / M;

            acc = acc * BASE + mat[i][j] + OFFSET;
        }

        packed_arr[START + FULL_GROUPS] = acc;
    }
}
comptime fn hash_all(W1: SSFP_MATRIX, W2: SSFP_MATRIX, W3: SSFP_MATRIX, b1: SSFP_MATRIX, b2: SSFP_MATRIX, b3:SSFP_MATRIX) -> CtString{
    let packed_arr: CtString = ssfp_context::append_ctx_MutVar(f"[Field; {PACKED_TOTAL}]", f"[0; {PACKED_TOTAL}]");
    
    let (mat_W1, mat_W2, mat_W3): (CtString, CtString, CtString) = (W1.mat, W2.mat, W3.mat);
    let (mat_B1, mat_B2, mat_B3): (CtString, CtString, CtString) = (b1.mat, b2.mat, b3.mat);
    let (BITS_W1, BITS_W2, BITS_W3): (u32, u32, u32) = (lvl_bits(W1.lvl), lvl_bits(W2.lvl), lvl_bits(W3.lvl));
    let (BITS_B1, BITS_B2, BITS_B3): (u32, u32, u32) = (lvl_bits(b1.lvl), lvl_bits(b2.lvl), lvl_bits(b3.lvl));
    let mut idx: u32 = 0;
    ssfp_context::append_ctx_code(f"pack::<{INPUT_SIZE},{HIDDEN_SIZE_1},{idx},{BITS_W1},{K_W1}>(&mut {packed_arr},{mat_W1});\n");
    idx += packed_len(INPUT_SIZE, HIDDEN_SIZE_1, W1.lvl);
    ssfp_context::append_ctx_code(f"pack::<{HIDDEN_SIZE_1},{HIDDEN_SIZE_2},{idx},{BITS_W2},{K_OTHERS}>(&mut {packed_arr},{mat_W2});\n");
    idx += packed_len(HIDDEN_SIZE_1, HIDDEN_SIZE_2, W2.lvl);
    ssfp_context::append_ctx_code(f"pack::<{HIDDEN_SIZE_2},{OUTPUT_SIZE},{idx},{BITS_W3},{K_OTHERS}>(&mut {packed_arr},{mat_W3});\n");
    idx += packed_len(HIDDEN_SIZE_2, OUTPUT_SIZE, W3.lvl);
    ssfp_context::append_ctx_code(f"pack::<1,{HIDDEN_SIZE_1},{idx},{BITS_B1},{K_OTHERS}>(&mut {packed_arr},{mat_B1});\n");
    idx += packed_len(1, HIDDEN_SIZE_1, b1.lvl);
    ssfp_context::append_ctx_code(f"pack::<1,{HIDDEN_SIZE_2},{idx},{BITS_B2},{K_OTHERS}>(&mut {packed_arr},{mat_B2});\n");
    idx += packed_len(1, HIDDEN_SIZE_2, b2.lvl);
    ssfp_context::append_ctx_code(f"pack::<1,{OUTPUT_SIZE},{idx},{BITS_B3},{K_OTHERS}>(&mut {packed_arr},{mat_B3});\n");

    let h = ssfp_context::append_ctx_var_Field(f"poseidon::poseidon2::Poseidon2::hash({packed_arr}, {PACKED_TOTAL})");
    h
}



fn main(
    data_train: [[Field; INPUT_SIZE]; BATCH_SIZE],
    labels_train: [Field; BATCH_SIZE],
    W1_raw: [[Field; HIDDEN_SIZE_1]; INPUT_SIZE],
    W2_raw: [[Field; HIDDEN_SIZE_2]; HIDDEN_SIZE_1],
    W3_raw: [[Field; OUTPUT_SIZE]; HIDDEN_SIZE_2],
    b1_raw: [Field; HIDDEN_SIZE_1],
    b2_raw: [Field; HIDDEN_SIZE_2],
    b3_raw: [Field; OUTPUT_SIZE]
) -> pub [Field; 3] {
    train_batch(data_train, labels_train, W1_raw, W2_raw, W3_raw, b1_raw, b2_raw, b3_raw)
}
