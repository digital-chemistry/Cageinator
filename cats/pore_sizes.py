#!/usr/bin/env python3
"""CATS v2: estimate the largest accessible pore of each assembly.

Input:
    ``--dir PROJECT_DIR`` containing ``assemblies/``.
Outputs:
    ``evaluation/pore_sizes/results.json`` and
    ``evaluation/pore_sizes/pore_size_distribution.png``.
Method:
    For assemblies with at least four Pd atoms, candidate windows are the
    triangular faces of the convex hull of the Pd coordinates.  The clearance
    at each face center is measured against atom-centered covalent radii.  A
    radial window search is used for two-Pd M2L4 assemblies.  Distributions are
    shown as Tukey boxplots grouped by assembly model.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull, QhullError

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


ANALYSIS_NAME = "pore_sizes"
FIGURE_NAME = "pore_size_distribution.png"

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


def two_node_window_radii(
    coordinates: np.ndarray,
    radii: np.ndarray,
    pd_coordinates: np.ndarray,
) -> list[float]:
    midpoint = pd_coordinates.mean(axis=0)
    axis = pd_coordinates[1] - pd_coordinates[0]
    axis /= np.linalg.norm(axis)

    trial_axis = np.asarray([1.0, 0.0, 0.0])
    if abs(np.dot(trial_axis, axis)) > 0.9:
        trial_axis = np.asarray([0.0, 1.0, 0.0])
    plane_axis_1 = np.cross(axis, trial_axis)
    plane_axis_1 /= np.linalg.norm(plane_axis_1)
    plane_axis_2 = np.cross(axis, plane_axis_1)

    vectors = coordinates - midpoint
    projected_radii = np.sqrt(
        np.square(vectors @ plane_axis_1) + np.square(vectors @ plane_axis_2)
    )
    ray_limit = float(np.max(projected_radii) + 4.0)

    bottlenecks = []
    radial_samples = np.linspace(0.0, ray_limit, 100)
    for angle in np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False):
        direction = np.cos(angle) * plane_axis_1 + np.sin(angle) * plane_axis_2
        ray_points = midpoint + radial_samples[:, np.newaxis] * direction
        distances = np.linalg.norm(
            ray_points[:, np.newaxis, :] - coordinates[np.newaxis, :, :],
            axis=-1,
        )
        minimum_clearance = np.min(distances - radii[np.newaxis, :], axis=1)
        bottlenecks.append(float(np.min(minimum_clearance)))

    padded = np.concatenate(([bottlenecks[-1]], bottlenecks, [bottlenecks[0]]))
    local_maxima = [
        padded[index]
        for index in range(1, len(padded) - 1)
        if padded[index] > 0.0
        and padded[index] > padded[index - 1]
        and padded[index] > padded[index + 1]
    ]
    return local_maxima or [max(bottlenecks)]


def calculate_pore_size(atoms: list[dict]) -> dict | None:
    if not atoms:
        return None

    coordinates = np.asarray([atom["xyz"] for atom in atoms], dtype=float)
    elements = [atom["el"] for atom in atoms]
    radii = np.asarray(
        [COVALENT_RADII.get(element, DEFAULT_COVALENT_RADIUS) for element in elements]
    )
    pd_indices = [index for index, element in enumerate(elements) if element == "Pd"]
    if len(pd_indices) < 2:
        return None

    pd_coordinates = coordinates[pd_indices]
    pore_radii = []
    if len(pd_coordinates) >= 4:
        try:
            hull = ConvexHull(pd_coordinates)
        except QhullError:
            return None
        for triangular_face in hull.simplices:
            face_center = pd_coordinates[triangular_face].mean(axis=0)
            clearances = np.linalg.norm(coordinates - face_center, axis=1) - radii
            pore_radii.append(float(np.min(clearances)))
    elif len(pd_coordinates) == 2:
        pore_radii = two_node_window_radii(coordinates, radii, pd_coordinates)

    if not pore_radii:
        return None

    maximum_radius = max(pore_radii)
    return {
        "max_pore_radius_angstrom": round(maximum_radius, 4),
        "max_pore_diameter_angstrom": round(2.0 * maximum_radius, 4),
        "average_candidate_pore_radius_angstrom": round(
            float(np.mean(pore_radii)),
            4,
        ),
        "candidate_windows_evaluated": len(pore_radii),
        "num_atoms": len(atoms),
    }


def generate_figure(results: dict, evaluation_dir: Path) -> Path:
    grouped = defaultdict(list)
    for cage_name, record in results.items():
        grouped[model_from_cage_name(cage_name)].append(
            record["max_pore_diameter_angstrom"]
        )

    configure_plot_style()
    figure, axis = plt.subplots(figsize=(7, 4.5))
    draw_model_boxplot(
        axis,
        grouped,
        r"Maximum pore diameter ($\mathrm{\AA}$)",
    )
    figure.tight_layout()
    figure_path = evaluation_dir / FIGURE_NAME
    figure.savefig(figure_path, bbox_inches="tight")
    plt.close(figure)
    return figure_path


def main() -> None:
    parser = build_parser("CATS: estimate maximum assembly pore diameters")
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
        pore = calculate_pore_size(atoms)
        metadata = parse_cage_name(cage_dir.name)
        if pore is None or metadata is None:
            continue
        pore["cage_name"] = cage_dir.name
        pore.update(metadata)
        pore["source_file"] = source_file
        results[cage_dir.name] = pore

    results_path = write_results(results, evaluation_dir)
    figure_path = generate_figure(results, evaluation_dir) if results else None
    report_finish(ANALYSIS_NAME, len(results), results_path, figure_path)


if __name__ == "__main__":
    main()
