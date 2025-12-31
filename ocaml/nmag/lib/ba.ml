(* This file previously had many type definitions for many
 ints and arrays, but they are no longer needed.*)
type c_layout = Bigarray.c_layout
let c_layout = Bigarray.c_layout
type matrix = (float, Bigarray.float64_elt, c_layout) Bigarray.Array2.t

(* MODERN INTRINSICS 
   These replace the old C-bindings. The "%" tells OCaml to use 
   built-in compiler primitives. These skip bounds checks.
*)
(* Unsafe access for Bigarray Array1 (rows) *)
external ba_unsafe_get_1 : (float, Bigarray.float64_elt, c_layout) Bigarray.Array1.t -> int -> float 
  = "%caml_ba_unsafe_ref_1"

let create_matrix rows cols : matrix = 
  Bigarray.Array2.create Bigarray.float64 c_layout rows cols

let matrix_x_vec ?store_result (mx : matrix) (v : float array) =
  let rows = Bigarray.Array2.dim1 mx in
  let cols = Bigarray.Array2.dim2 mx in

  if cols <> Array.length v then
    invalid_arg (Printf.sprintf "Dimension mismatch: matrix cols %d vs vector len %d" cols (Array.length v));

  let result =
    match store_result with
    | None -> Array.make rows 0.0
    | Some buf -> 
        if Array.length buf <> rows then invalid_arg "Result buffer size mismatch";
        buf
  in

  for i = 0 to rows - 1 do
    (* Slice the row (fast view creation) *)
    let row = Bigarray.Array2.slice_left mx i in
    let sum = ref 0.0 in
    
    (* Inner dot-product loop *)
    for j = 0 to cols - 1 do
      let r_val = ba_unsafe_get_1 row j in
      let v_val = Array.unsafe_get v j in
      sum := !sum +. (r_val *. v_val)
    done;
    
    Array.unsafe_set result i !sum
  done;
  result
