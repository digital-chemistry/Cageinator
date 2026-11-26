# CATS: Cageinator Analysis Toolset

Five scripts that analyse the optimised assemblies written by Cageinator. They read `assemblies/` of a project directory
(the optimised cage is `xtb_opt.xyz` in each cage folder) and write `results.json` and a figure to `evaluation/<analysis>/`.

| Command | Analysis |
|---|---|
| `python -m cats.bond_analysis --dir PROJECT_DIR` | Pd–N distances, cis and trans N–Pd–N angles and the N<sub>4</sub> dihedral at every Pd centre |
| `python -m cats.cage_sizes --dir PROJECT_DIR` | maximum outer diameter |
| `python -m cats.pore_sizes --dir PROJECT_DIR` | largest pore (aperture) diameter |
| `python -m cats.vacancy_size --dir PROJECT_DIR` | internal cavity diameter |
| `python -m cats.electronic_energies --dir PROJECT_DIR` | GFN-FF/GBSA(water) energies of the assembly models relative to each other, and their relation to the Pd coordination geometry |

Run them from the repository root. The data of the manuscript are in `data/`, so `--dir data` reproduces the files in `data/evaluation/`.
The scripts need Python 3.10+ with `numpy`, `scipy` and `matplotlib` (`pip install .[cats]`).
