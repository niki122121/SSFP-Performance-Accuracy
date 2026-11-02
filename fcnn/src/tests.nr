// uncoment for basic tests
// for deeper test of full run, in main.nr there is a TEST_MODE and TEST_PRINT_ALL. 
// When enabled we can print all updated weights, biases, loss... to console for manual inspection
// It is recomended to print into a separate file as it can generate a lot of text.





// use ssfp::{SSFP_MATRIX};
// use ssfp::{SSFP_CTX_INIT, SSFP_CTX_SOLVE, SSFP_PUSH_OUTPUT};

// use super::{relu, relu_der, softmax, validate_and_adjust_labels_rt, select_correct_probs_rt};
// use super::{SSFP_MD_UNKNOWN, OUTPUT_SIZE, OUTPUT_SIZE_RT, BATCH_SIZE_RT};

// // Small helpers that execute the actual SSFP_MATRIX operations.

// fn fcnn_relu_probe(x: [Field; 5]) -> [Field; 5] {
//     let res = comptime {
//         SSFP_CTX_INIT();
//         let mat: SSFP_MATRIX = SSFP_MATRIX::new_from_arr(1, 5, 1, SSFP_MD_UNKNOWN, "x");
//         let out: SSFP_MATRIX = relu(mat);
//         SSFP_PUSH_OUTPUT(out.get_row(0));
//         SSFP_CTX_SOLVE(false)
//     };
//     res
// }

// fn fcnn_relu_der_probe(x: [Field; 5]) -> [Field; 5] {
//     let res = comptime {
//         SSFP_CTX_INIT();
//         let mat: SSFP_MATRIX = SSFP_MATRIX::new_from_arr(1, 5, 1, SSFP_MD_UNKNOWN, "x");
//         let out: SSFP_MATRIX = relu_der(mat);
//         SSFP_PUSH_OUTPUT(out.get_row(0));
//         SSFP_CTX_SOLVE(false)
//     };
//     res
// }

// fn fcnn_softmax_probe(logits: [Field; OUTPUT_SIZE]) -> [Field; OUTPUT_SIZE] {
//     let res = comptime {
//         SSFP_CTX_INIT();
//         let mat: SSFP_MATRIX = SSFP_MATRIX::new_from_arr(1, OUTPUT_SIZE, 1, SSFP_MD_UNKNOWN, "logits");
//         let out: SSFP_MATRIX = softmax::<1>(mat);
//         SSFP_PUSH_OUTPUT(out.get_row(0));
//         SSFP_CTX_SOLVE(false)
//     };
//     res
// }

// // ----------------------------------------------------------
// // TEST 1: ReLU with negative, zero, and positive inputs
// // ----------------------------------------------------------
// #[test]
// fn fcnn_test_relu() {
//     let x: [Field; 5] = [-17, -1, 0, 3, 17];
//     let expected: [Field; 5] = [0, 0, 0, 3, 17];

//     let result = fcnn_relu_probe(x);

//     for i in 0..5 {
//         assert(result[i] == expected[i]);
//     }
// }

// // ----------------------------------------------------------
// // TEST 2: ReLU derivative
// // ----------------------------------------------------------
// #[test]
// fn fcnn_test_relu_derivative() {
//     let x: [Field; 5] = [-17, -1, 0, 3, 17];

//     // Current SSFP behavior: derivative at zero is 1.
//     let expected: [Field; 5] = [0, 0, 1, 1, 1];

//     let result = fcnn_relu_der_probe(x);

//     for i in 0..5 {
//         assert(result[i] == expected[i]);
//     }
// }

// // ----------------------------------------------------------
// // TEST 3: One-hot encoding of labels
// // ----------------------------------------------------------
// #[test]
// fn fcnn_test_one_hot() {
//     let labels: [Field; BATCH_SIZE_RT] = [0, 3, 7, 9];
//     let encoded = validate_and_adjust_labels_rt::<BATCH_SIZE_RT>(labels);

//     for i in 0..BATCH_SIZE_RT {
//         for j in 0..OUTPUT_SIZE_RT {
//             let expected: Field = if j == labels[i] as u32 { 1 } else { 0 };
//             assert(encoded[i][j] == expected);
//         }
//     }
// }

// // ----------------------------------------------------------
// // TEST 4: An invalid label must fail
// // ----------------------------------------------------------
// #[test(should_fail)]
// fn fcnn_test_invalid_label() {
//     let labels: [Field; BATCH_SIZE_RT] = [0, 3, 7, OUTPUT_SIZE_RT as Field];
//     let _ = validate_and_adjust_labels_rt::<BATCH_SIZE_RT>(labels);
// }

// // ----------------------------------------------------------
// // TEST 5: Select probabilities for correct classes
// // ----------------------------------------------------------
// #[test]
// fn fcnn_test_select_correct_probabilities() {
//     let labels: [Field; BATCH_SIZE_RT] = [0, 3, 7, 9];
//     let encoded = validate_and_adjust_labels_rt::<BATCH_SIZE_RT>(labels);

//     let mut probabilities: [[Field; OUTPUT_SIZE_RT]; BATCH_SIZE_RT] = [[0; OUTPUT_SIZE_RT]; BATCH_SIZE_RT];

//     probabilities[0][0] = 11;
//     probabilities[1][3] = 22;
//     probabilities[2][7] = 33;
//     probabilities[3][9] = 44;

//     let selected = select_correct_probs_rt(probabilities, encoded);

//     assert(selected[0] == 11);
//     assert(selected[1] == 22);
//     assert(selected[2] == 33);
//     assert(selected[3] == 44);
// }

// // ----------------------------------------------------------
// // TEST 6: Equal logits must produce equal probabilities
// // ----------------------------------------------------------
// #[test]
// fn fcnn_test_softmax_uniform() {
//     let logits: [Field; OUTPUT_SIZE_RT] = [0; OUTPUT_SIZE_RT];
//     let probabilities = fcnn_softmax_probe(logits);

//     assert(probabilities[0] != 0);

//     for j in 1..OUTPUT_SIZE_RT {
//         assert(probabilities[j] == probabilities[0]);
//     }

//     let aux = probabilities[0];
//     println(f"Uniform softmax probability (raw): {aux}");
// }
