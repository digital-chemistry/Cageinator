# Data: Pd(II) coordination cages built with Cageinator

53 ditopic linkers combined with a square-planar Pd(II) node in four assembly models give 212 node–linker–model combinations.
187 of them passed the geometric screening of Cageinator and were optimised with GFN-FF; their structures are in `assemblies/`.
The inputs, the optimised and constructed geometries, and the results of the CATS analyses (`../cats/`) are the data of the manuscript
*High-Throughput Construction and Optimisation of Coordination Cages* (J. Leodolter, A. Kondinski).

## Contents

| Path | Content |
|---|---|
| `nodes/PdII.json` | the square-planar Pd(II) node used for all cages |
| `linkers/*.xyz` | the 53 ditopic linkers (input coordinates) |
| `assemblies/<cage>/` | one folder per cage, see below |
| `optimised_structures.csv` | index of the 187 cages: linker, assembly model, charge, number of atoms, energy, file paths |
| `evaluation/` | results (`results.json`) and figures of the five CATS analyses |

The linkers and their codes (C1-1, C2-1, …) are those of the meta-analysis by Tateishi et al., *Coord. Chem. Rev.* **2022**, *467*, 214612,
https://doi.org/10.1016/j.ccr.2022.214612.

## Assembly models

| Assembly model | Name in the cage folders | Cages |
|---|---|---|
| (4-planar)<sub>2</sub>(2-bent)<sub>4</sub>-D<sub>4h</sub> (lantern, M<sub>2</sub>L<sub>4</sub>) | `M2L4_lantern` | 42 |
| (4-planar)<sub>6</sub>(2-bent)<sub>12</sub>-O<sub>h</sub> (octahedron, M<sub>6</sub>L<sub>12</sub>) | `M6L12_octaeder` | 46 |
| (4-planar)<sub>12</sub>(2-bent)<sub>24</sub>-O<sub>h</sub> (cuboctahedron, M<sub>12</sub>L<sub>24</sub>) | `M12L24_cuboctaeder` | 50 |
| (4-planar)<sub>12</sub>(2-bent)<sub>24</sub>-D<sub>3h</sub> (cuboctahedron, M<sub>12</sub>L<sub>24</sub>) | `M12L24_cuboctaeder_d3h` | 49 |

## Cage folders

A folder is named `<node>__<linker>__<stoichiometry_model>__Q<charge>`, for example `PdII__C1-10__M12L24_cuboctaeder__Q+24`:

- `<name>.mol`: the structure as constructed by Cageinator (starting geometry of the optimisation)
- `xtb_opt.xyz`: the GFN-FF optimised structure; the second line holds the energy in hartree (`energy: …`)
- `xtb_opt.log`: the xTB output of the GFN-FF/GBSA(water) single-point calculation on `xtb_opt.xyz`

The geometries were optimised in the gas phase with GFN-FF (xTB 6.7.1, `--opt tight`, at most 1000 cycles). The energy in `xtb_opt.xyz`
is the GFN-FF/GBSA(water) total energy of the optimised structure (`--gfnff --gbsa water`); it is the quantity that CATS compares between
assembly models. `optimised_structures.csv` lists the same energies.

## Combinations without a cage

The 25 combinations that did not pass the screening (steric clashes in the lantern, atoms entering the cavity in the larger cages) have no folder:

| Assembly model | Linkers |
|---|---|
| lantern | C2-3, C2-4, C2-5, C2-6, C2-7, C2-9, C2-10, C2-11, C2-12, C2-13, C2-14 |
| octahedron | C2-3, C2-4, C2-5, C2-10, C2-11, C2-13, C2-14 |
| cuboctahedron (O<sub>h</sub>) | C2-4, C2-5, C2-14 |
| cuboctahedron (D<sub>3h</sub>) | C1-15, C2-4, C2-5, C2-14 |

A rebuild with the code of this repository reproduces all 187 cages. It also builds C1-15 as the D<sub>3h</sub> cuboctahedron
(2868 atoms), which is not part of the deposited data and the analyses.

## Reproducing the analyses

```bash
python -m cats.cage_sizes --dir data
```

writes `data/evaluation/cage_sizes/results.json` and the figure. Run the commands from the repository root (see `../cats/README.md`).
To rebuild structures instead, use the inputs in `nodes/` and `linkers/`:

```bash
cageinator --nodes data/nodes --linkers data/linkers --out my_out --shape cuboctaeder
```
