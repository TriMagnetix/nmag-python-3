import ocaml
import inspect

ocaml.require('nmag')

from ocaml import Nmag

# print(Nmag.Example.add(3,5))

print(inspect.getmembers(Nmag.Example, inspect.isroutine))

print(Nmag.Example.create_matrix)
