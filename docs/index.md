# Nmag for Python 3

Nmag is a Python interface for finite-element micromagnetic simulations. This
rewrite runs on modern Python, reads legacy Nmesh files and common mesh formats,
and provides static fields, adaptive magnetization dynamics, relaxation,
checkpointing, and structured NDT/HDF5 output.

[Install Nmag](getting-started/installation.md){ .md-button .md-button--primary }
[Run the quickstart](getting-started/quickstart.md){ .md-button }

## Where to begin

- **New to the package?** Follow the [quickstart](getting-started/quickstart.md),
  then work through the [sphere tutorial](getting-started/sphere.md).
- **Building a simulation?** Read the [simulation workflow](guides/simulation-workflow.md)
  and the guides for [meshes](guides/meshes.md),
  [materials](guides/materials.md), and [output](guides/output.md).
- **Moving an old script?** Start with the
  [legacy migration guide](guides/migration.md).
- **Looking up a call?** Use the curated
  [API reference](reference/simulation.md).

!!! warning "Current project status"

    The supported workflows are usable, but the rewrite does not implement
    every feature of Nmag 0.2. Validate a new geometry or material model before
    relying on production results, and check the
    [supported scope](guides/supported-scope.md) before porting a legacy study.

## A Python simulation, end to end

An Nmag program follows a small set of explicit steps:

1. load or construct a tetrahedral mesh;
2. define magnetic materials with SI-valued parameters;
3. create a simulation and map mesh regions to materials;
4. set magnetization and applied fields;
5. calculate fields, advance time, or relax the state; and
6. save averages and spatial fields for analysis.

The documentation concentrates on that user workflow. Internal solver design,
development investigations, and validation machinery are intentionally not part
of this manual.

## Project lineage

Nmag was originally developed at the University of Southampton. The historical
[Nmag 0.2 manual](https://nmag.readthedocs.io/en/latest/) remains useful for
micromagnetic background, but its Python 2 syntax, command-line tools, solver
options, and some supported physics do not describe this rewrite. The original
source is available from
[nmag-project/nmag-src](https://github.com/nmag-project/nmag-src).
