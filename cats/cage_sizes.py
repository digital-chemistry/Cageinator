#!/usr/bin/env python3
"""CATS v2: calculate the maximum outer diameter of each assembly.

Input:
    ``--dir PROJECT_DIR`` containing ``assemblies/``.
Outputs:
    ``evaluation/cage_sizes/results.json`` and
    ``evaluation/cage_sizes/outer_size_distribution.png``.
Method:
    The outer diameter is the largest atom-pair separation after adding the
    covalent radius of both limiting atoms.  Distributions are shown as Tukey
    boxplots grouped by assembly model.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from ._shared import (
        build_parser,
        configure_plot_style,
        draw_model_boxplot,
        load_assembly_geometry,
        model_from_cage_name,
        parse_cage_name,
        report_finish,
        report_start,
        resolve_directories,
        write_results,
    )
except ImportError:
    from _shared import (
        build_parser,
        configure_plot_style,
        draw_model_boxplot,
        load_assembly_geometry,
        model_from_cage_name,
        parse_cage_name,
        report_finish,
        report_start,
        resolve_directories,
        write_results,
    )


ANALYSIS_NAME = "cage_sizes"
FIGURE_NAME = "outer_size_distribution.png"

COVALENT_RADII = {
    "H": 0.31,
    "B": 0.85,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
    "Pd": 1.39,
    "Cu": 1.32,
    "Zn": 1.22,
    "Pt": 1.36,
    "Au": 1.36,
    "Ag": 1.45,
}
DEFAULT_COVALENT_RADIUS = 0.75


def calculate_outer_size(atoms: list[dict]) -> dict | None:
    if not atoms:
        return None

    coordinates = np.asarray([atom["xyz"] for atom in atoms], dtype=float)
    elements = [atom["el"] for atom in atoms]
    radii = np.asarray(
        [COVALENT_RADII.get(element, DEFAULT_COVALENT_RADIUS) for element in elements]
    )

    pair_distances = np.linalg.norm(
        coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :],
        axis=-1,
    )
    outer_distances = pair_distances + radii[:, np.newaxis] + radii[np.newaxis, :]
    limiting_indices = np.unravel_index(np.argmax(outer_distances), outer_distances.shape)
    diameter = float(outer_distances[limiting_indices])

    return {
        "outer_diameter_angstrom": round(diameter, 4),
        "outer_radius_angstrom": round(diameter / 2.0, 4),
        "limiting_atom_1": {
            "index": int(limiting_indices[0]),
            "element": elements[limiting_indices[0]],
        },
        "limiting_atom_2": {
            "index": int(limiting_indices[1]),
            "element": elements[limiting_indices[1]],
        },
        "num_atoms": len(atoms),
    }


def generate_figure(results: dict, evaluation_dir: Path) -> Path:
    grouped = defaultdict(list)
    for cage_name, record in results.items():
        grouped[model_from_cage_name(cage_name)].append(
            record["outer_diameter_angstrom"]
        )

    configure_plot_style()
    figure, axis = plt.subplots(figsize=(7, 4.5))
    draw_model_boxplot(
        axis,
        grouped,
        r"Outer assembly diameter ($\mathrm{\AA}$)",
    )
    figure.tight_layout()
    figure_path = evaluation_dir / FIGURE_NAME
    figure.savefig(figure_path, bbox_inches="tight")
    plt.close(figure)
    return figure_path


def main() -> None:
    parser = build_parser("CATS: calculate outer assembly diameters")
    args = parser.parse_args()
    _, assemblies_dir, evaluation_dir = resolve_directories(args.dir, ANALYSIS_NAME)
    report_start(ANALYSIS_NAME, assemblies_dir)

    results = {}
    for cage_dir in sorted(assemblies_dir.iterdir()):
        if not cage_dir.is_dir():
            continue
        atoms, source_file = load_assembly_geometry(
            cage_dir,
            require_optimized=True,
        )
        size = calculate_outer_size(atoms)
        metadata = parse_cage_name(cage_dir.name)
        if size is None or metadata is None:
            continue
        size["cage_name"] = cage_dir.name
        size.update(metadata)
        size["source_file"] = source_file
        results[cage_dir.name] = size

    results_path = write_results(results, evaluation_dir)
    figure_path = generate_figure(results, evaluation_dir) if results else None
    report_finish(ANALYSIS_NAME, len(results), results_path, figure_path)


if __name__ == "__main__":
    main()
