open Bigarray
(* open Py *)
(* type c_layout = Bigarray.c_layout
let c_layout = Bigarray.c_layout *)
type matrix = (float, Bigarray.float64_elt, c_layout) Bigarray.Array2.t
let create_matrix rows cols: matrix = 
  let ba = Array2.create Float64 c_layout rows cols in
  ba

let add x y = x + y

let () =
  print_endline ( string_of_float ((create_matrix 3 5).{0,0}))
