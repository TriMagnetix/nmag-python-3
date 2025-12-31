(* --- Test Setup --- *)
(* Helper to create a Bigarray matrix from a list of lists.
   Defined locally so we don't pollute the main library. *)
let matrix_of_list lst =
  let rows = List.length lst in
  let cols = if rows = 0 then 0 else List.length (List.hd lst) in
  (* Create standard Bigarray with C layout *)
  let m = Bigarray.Array2.create Bigarray.float64 Bigarray.c_layout rows cols in
  List.iteri (fun r_idx row ->
    List.iteri (fun c_idx v ->
      Bigarray.Array2.set m r_idx c_idx v
    ) row
  ) lst;
  m

(* Define a float checker with a small tolerance (epsilon) for floating point math *)
let epsilon = 1e-6
let test_float = Alcotest.float epsilon
let test_float_array = Alcotest.array test_float

(* --- Test Cases --- *)

let test_basic_mult () =
  let m = matrix_of_list [[1.0; 2.0]; [3.0; 4.0]] in
  let v = [| 1.0; 1.0 |] in 
  (* Call the library function *)
  let res = Nmag.Ba.matrix_x_vec m v in
  (* Expected: [1*1 + 2*1; 3*1 + 4*1] = [3.0; 7.0] *)
  Alcotest.(check test_float_array) "2x2 multiplication correct" [| 3.0; 7.0 |] res

let test_identity () =
  let m = matrix_of_list [[1.0; 0.0]; [0.0; 1.0]] in
  let v = [| 10.0; 20.0 |] in
  let res = Nmag.Ba.matrix_x_vec m v in
  Alcotest.(check test_float_array) "Identity preserves vector" v res

let test_rectangular () =
  (* 3x2 Matrix multiplied by 2-element vector 
     | 1 2 |   | 1 |   | 5 |
     | 3 4 | x | 2 | = | 11|
     | 5 6 |           | 17|
  *)
  let m = matrix_of_list [[1.0; 2.0]; [3.0; 4.0]; [5.0; 6.0]] in
  let v = [| 1.0; 2.0 |] in
  let res = Nmag.Ba.matrix_x_vec m v in
  let expected = [| 5.0; 11.0; 17.0 |] in
  Alcotest.(check test_float_array) "3x2 multiplication correct" expected res

let test_buffer_usage () =
  let m = matrix_of_list [[1.0; 2.0]; [3.0; 4.0]] in
  let v = [| 1.0; 1.0 |] in
  let buf = Array.make 2 0.0 in
  (* Pass the buffer to store_result *)
  let _ = Nmag.Ba.matrix_x_vec ~store_result:buf m v in
  let expected = [| 3.0; 7.0 |] in
  Alcotest.(check test_float_array) "Buffer was populated correctly" expected buf

let test_dim_mismatch () =
  let m = matrix_of_list [[1.0; 2.0]] in (* 1x2 *)
  let v = [| 1.0; 2.0; 3.0 |] in         (* Length 3 *)
  
  try
    let _ = Nmag.Ba.matrix_x_vec m v in
    Alcotest.fail "Should have raised Invalid_argument"
  with Invalid_argument _ -> 
    Alcotest.(check pass) "Raised Invalid_argument correctly" () ()

(* --- Test Runner --- *)

let () =
  let open Alcotest in
  run "Nmag_Ba" [
    "matrix_x_vec", [
      test_case "Basic 2x2"     `Quick test_basic_mult;
      test_case "Identity"      `Quick test_identity;
      test_case "Rectangular"   `Quick test_rectangular;
      test_case "Buffer Write"  `Quick test_buffer_usage;
    ];
    "errors", [
      test_case "Dimension Mismatch" `Quick test_dim_mismatch;
    ]
  ]