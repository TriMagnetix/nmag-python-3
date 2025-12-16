import ocaml

ocaml.require('nmag')

from ocaml import Example

sum_val = Example.add(3, 7)

print(f"Result: {sum_val}")