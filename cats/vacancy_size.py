#!/usr/bin/env python3
"""CATS v2: calculate the largest internal cavity sphere of each assembly.

Input:
    ``--dir PROJECT_DIR`` containing ``assemblies/``; ``--static`` optionally
    fixes the sphere center at the geometric centroid.
Outputs:
    ``evaluation/vacancy_size/results.json`` and
    ``evaluation/vacancy_size/vacancy_distribution.png``.
Method:
    Nelder--Mead maximizes the minimum clearance from the trial center to all
    atom-centered van der Waals surfaces.  Distributions are shown as Tukey
    boxplots grouped by assembly model.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

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


ANALYSIS_NAME = "vacancy_size"
FIGURE_NAME = "vacancy_distribution.png"

VAN_DER_WAALS_RADII = {
    "H": 1.20,
    "He": 1.40,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "Ne": 1.54,
    "Si": 2.10,
    "P": 1.80,
    "S": 1.80,
    "Cl": 1.75,
    "Ar": 1.88,
    "Pd": 1.63,
    "Cu": 1.40,
    "Zn": 1.39,
    "Pt": 1.75,
    "Au": 1.66,
    "Ag": 1.72,
}
DEFAULT_VAN_DER_WAALS_RADIUS = 1.50


def calculate_vacancy(atoms: list[dict], *, optimize_center: bool = True) -> dict | None:
    if not atoms:
        return None

    coordinates = np.asarray([atom["xyz"] for atom in atoms], dtype=float)
    elements = [atom["el"] for atom in atoms]
    radii = np.asarray(
        [
            VAN_DER_WAALS_RADII.get(element, DEFAULT_VAN_DER_WAALS_RADIUS)
            for element in elements
        ]
    )
    centroid = coordinates.mean(axis=0)

    def negative_clearance(center: np.ndarray) -> float:
        clearances = np.linalg.norm(coordinates - center, axis=1) - radii
        return -float(np.min(clearances))

    optimization_success = True
    if optimize_center:
        optimization = minimize(
            negative_clearance,
            centroid,
            method="Nelder-Mead",
            options={"xatol": 1.0e-4, "fatol": 1.0e-4},
        )
        best_center = optimization.x
        maximum_radius = -float(optimization.fun)
        optimization_success = bool(optimization.success)
    else:
        best_center = centroid
        maximum_radius = -negative_clearance(centroid)

    final_distances = np.linalg.norm(coordinates - best_center, axis=1)
    final_clearances = final_distances - radii
    limiting_index = int(np.argmin(final_clearances))

    return {
        "centroid_initial": [float(value) for value in centroid],
        "optimized_center": [float(value) for value in best_center],
        "center_shift_angstrom": round(
            float(np.linalg.norm(best_center - centroid)),
            4,
        ),
        "vacancy_radius_angstrom": round(maximum_radius, 4),
        "vacancy_diameter_angstrom": round(2.0 * maximum_radius, 4),
        "limiting_atom_element": elements[limiting_index],
        "limiting_atom_index": limiting_index,
        "limiting_atom_distance_angstrom": round(
            float(final_distances[limiting_index]),
            4,
        ),
        "center_optimization_enabled": optimize_center,
        "center_optimization_success": optimization_success,
        "num_atoms": len(atoms),
    }


def generate_figure(results: dict, evaluation_dir: Path) -> Path:
    grouped = defaultdict(list)
    for cage_name, record in results.items():
        grouped[model_from_cage_name(cage_name)].append(
            record["vacancy_diameter_angstrom"]
        )

    configure_plot_style()
    figure, axis = plt.subplots(figsize=(7, 4.5))
    draw_model_boxplot(
        axis,
        grouped,
        r"Internal cavity diameter ($\mathrm{\AA}$)",
    )
    figure.tight_layout()
    figure_path = evaluation_dir / FIGURE_NAME
    figure.savefig(figure_path, bbox_inches="tight")
    plt.close(figure)
    return figure_path


def main() -> None:
    parser = build_parser(
        "CATS: calculate internal assembly cavity diameters",
        static_option=True,
    )
    args = parser.parse_args()
    _, assemblies_dir, evaluation_dir = resolve_directories(args.dir, ANALYSIS_NAME)
    report_start(ANALYSIS_NAME, assemblies_dir)

    results = {}
    for cage_dir in sorted(assemblies_dir.iterdir()):
        if not cage_dir.is_dir():
            continue
        atoms, source_file = load_assembly_geometry(cage_dir)
        vacancy = calculate_vacancy(atoms, optimize_center=not args.static)
        metadata = parse_cage_name(cage_dir.name)
        if vacancy is None or metadata is None:
            continue
        vacancy["cage_name"] = cage_dir.name
        vacancy.update(metadata)
        vacancy["source_file"] = source_file
        results[cage_dir.name] = vacancy

    results_path = write_results(results, evaluation_dir)
    figure_path = generate_figure(results, evaluation_dir) if results else None
    report_finish(ANALYSIS_NAME, len(results), results_path, figure_path)


if __name__ == "__main__":
    main()
