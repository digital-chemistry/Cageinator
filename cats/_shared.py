"""Shared input, naming, geometry, and plotting utilities for CATS v2.

All five Cageinator Analysis Toolset (CATS) scripts accept the same project
root through ``--dir``.  The root must contain ``assemblies/``; each script
writes ``results.json`` and one publication figure below ``evaluation/``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


MODEL_ORDER = (
    "lantern",
    "octaeder",
    "cuboctaeder",
    "cuboctaeder_d3h",
)

MODEL_LABELS = {
    "lantern": r"$M_2L_4$",
    "octaeder": r"$M_6L_{12}$",
    "cuboctaeder": r"$M_{12}L_{24}$ ($O_h$)",
    "cuboctaeder_d3h": r"$M_{12}L_{24}$ ($D_{3h}$)",
}

MODEL_TEXT = {
    "lantern": "M2L4",
    "octaeder": "M6L12",
    "cuboctaeder": "M12L24 (Oh)",
    "cuboctaeder_d3h": "M12L24 (D3h)",
}

MODEL_COLORS = {
    "lantern": "#B9DDEC",
    "octaeder": "#477AA5",
    "cuboctaeder": "#A78CC3",
    "cuboctaeder_d3h": "#E58E8E",
}


def model_from_cage_name(cage_name: str) -> str:
    """Return the internal assembly-model key encoded in a cage name."""
    parts = cage_name.split("__")
    if len(parts) < 3 or "_" not in parts[2]:
        return "unknown"
    return parts[2].split("_", 1)[1]


def parse_cage_name(cage_name: str) -> dict | None:
    """Parse node, linker, stoichiometry, and assembly model from a cage name."""
    parts = cage_name.split("__")
    if len(parts) < 3:
        return None

    match = re.fullmatch(r"M(\d+)L(\d+)_(.+)", parts[2])
    if match is None:
        return None

    metal_count = int(match.group(1))
    linker_count = int(match.group(2))
    model = match.group(3)
    return {
        "system_key": f"{parts[0]}__{parts[1]}",
        "node": parts[0],
        "linker": parts[1],
        "metal_count": metal_count,
        "linker_count": linker_count,
        "ml2_units": metal_count,
        "model_key": model,
        "assembly_model": MODEL_TEXT.get(model, model),
    }


def ordered_models(models: Iterable[str]) -> list[str]:
    """Order known models by nuclearity and append any unknown models."""
    present = set(models)
    known = [model for model in MODEL_ORDER if model in present]
    return known + sorted(present.difference(MODEL_ORDER))


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model.replace("_", " "))


def model_color(model: str) -> str:
    return MODEL_COLORS.get(model, "#777777")


def build_parser(description: str, *, static_option: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dir",
        required=True,
        metavar="PROJECT_DIR",
        help="Cageinator project root containing assemblies/",
    )
    if static_option:
        parser.add_argument(
            "--static",
            action="store_true",
            help="Keep the cavity center at the geometric centroid",
        )
    return parser


def resolve_directories(project_dir: str, analysis_name: str) -> tuple[Path, Path, Path]:
    root = Path(project_dir).expanduser().resolve()
    assemblies_dir = root / "assemblies"
    if not assemblies_dir.is_dir():
        print(
            f"[CATS:error] Assemblies directory not found: {assemblies_dir}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    evaluation_dir = root / "evaluation" / analysis_name
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    return root, assemblies_dir, evaluation_dir


def write_results(results: dict, evaluation_dir: Path) -> Path:
    output_path = evaluation_dir / "results.json"
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def report_start(analysis_name: str, assemblies_dir: Path) -> None:
    print(f"[CATS:{analysis_name}] input: {assemblies_dir}")


def report_finish(
    analysis_name: str,
    result_count: int,
    results_path: Path,
    figure_path: Path | None,
) -> None:
    print(f"[CATS:{analysis_name}] analyzed: {result_count} assemblies")
    print(f"[CATS:{analysis_name}] results: {results_path}")
    if figure_path is not None:
        print(f"[CATS:{analysis_name}] figure: {figure_path}")


def parse_xyz(path: Path) -> list[dict]:
    """Read element symbols and Cartesian coordinates from an XYZ file."""
    lines = [line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()]
    if not lines:
        return []

    try:
        atom_count = int(lines[0].strip())
        coordinate_lines = lines[2 : 2 + atom_count]
    except ValueError:
        coordinate_lines = [line for line in lines if line.strip()]

    atoms = []
    for line in coordinate_lines:
        fields = line.split()
        if len(fields) < 4:
            continue
        try:
            xyz = np.asarray([float(value) for value in fields[1:4]], dtype=float)
        except ValueError:
            continue
        atoms.append({"el": fields[0], "xyz": xyz})
    return atoms


def parse_mol_v3000(path: Path) -> list[dict]:
    """Read the atom block of the V3000 molfile written by Cageinator."""
    atoms = []
    in_atom_block = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if "BEGIN ATOM" in line:
            in_atom_block = True
            continue
        if "END ATOM" in line:
            in_atom_block = False
            continue
        if not in_atom_block:
            continue

        fields = line.split()
        if len(fields) < 7 or fields[1] != "V30":
            continue
        try:
            xyz = np.asarray([float(value) for value in fields[4:7]], dtype=float)
        except ValueError:
            continue
        atoms.append({"el": fields[3], "xyz": xyz})
    return atoms


def load_assembly_geometry(cage_dir: Path, *, require_optimized: bool = False) -> tuple[list[dict], str]:
    """Load optimized XYZ coordinates, with an optional assembled-mol fallback."""
    optimized_xyz = cage_dir / "xtb_opt.xyz"
    if optimized_xyz.is_file() and optimized_xyz.stat().st_size > 0:
        return parse_xyz(optimized_xyz), optimized_xyz.name

    if require_optimized:
        return [], ""

    assembled_mol = cage_dir / f"{cage_dir.name}.mol"
    if assembled_mol.is_file() and assembled_mol.stat().st_size > 0:
        return parse_mol_v3000(assembled_mol), assembled_mol.name
    return [], ""


def angle_degrees(point_a: np.ndarray, vertex: np.ndarray, point_b: np.ndarray) -> float:
    vector_a = point_a - vertex
    vector_b = point_b - vertex
    norm_product = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if norm_product <= 1.0e-15:
        return float("nan")
    cosine = np.clip(np.dot(vector_a, vector_b) / norm_product, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def dihedral_degrees(
    point_a: np.ndarray,
    point_b: np.ndarray,
    point_c: np.ndarray,
    point_d: np.ndarray,
) -> float:
    bond_1 = point_b - point_a
    bond_2 = point_c - point_b
    bond_3 = point_d - point_c
    normal_1 = np.cross(bond_1, bond_2)
    normal_2 = np.cross(bond_2, bond_3)

    norm_1 = np.linalg.norm(normal_1)
    norm_2 = np.linalg.norm(normal_2)
    norm_bond_2 = np.linalg.norm(bond_2)
    if min(norm_1, norm_2, norm_bond_2) <= 1.0e-15:
        return float("nan")

    normal_1 /= norm_1
    normal_2 /= norm_2
    auxiliary = np.cross(normal_1, bond_2 / norm_bond_2)
    return float(
        np.degrees(
            np.arctan2(np.dot(auxiliary, normal_2), np.dot(normal_1, normal_2))
        )
    )


def analyze_square_planar_pd(atoms: list[dict], cutoff: float = 2.5) -> dict | None:
    """Measure four-coordinate Pd--N environments in one assembly.

    The four nearest N atoms within ``cutoff`` are used for each Pd atom.  The
    two largest of the six N--Pd--N angles are classified as trans; the
    remaining four are classified as cis.
    """
    pd_indices = [index for index, atom in enumerate(atoms) if atom["el"] == "Pd"]
    n_indices = [index for index, atom in enumerate(atoms) if atom["el"] == "N"]
    if not pd_indices or len(n_indices) < 4:
        return None

    distances = []
    cis_angles = []
    trans_angles = []
    n4_dihedrals = []
    angular_residuals = []
    analyzed_pd = 0

    for pd_index in pd_indices:
        pd_position = atoms[pd_index]["xyz"]
        neighboring_n = sorted(
            (
                (
                    float(np.linalg.norm(atoms[n_index]["xyz"] - pd_position)),
                    atoms[n_index]["xyz"],
                )
                for n_index in n_indices
            ),
            key=lambda item: item[0],
        )
        neighboring_n = [item for item in neighboring_n if item[0] < cutoff][:4]
        if len(neighboring_n) != 4:
            continue

        analyzed_pd += 1
        distances.extend(distance for distance, _ in neighboring_n)
        n_positions = np.asarray([position for _, position in neighboring_n])

        six_angles = sorted(
            angle_degrees(n_positions[i], pd_position, n_positions[j])
            for i in range(4)
            for j in range(i + 1, 4)
        )
        local_cis = six_angles[:4]
        local_trans = six_angles[4:]
        cis_angles.extend(local_cis)
        trans_angles.extend(local_trans)
        angular_residuals.extend(angle - 90.0 for angle in local_cis)
        angular_residuals.extend(angle - 180.0 for angle in local_trans)

        center = n_positions.mean(axis=0)
        _, _, right_vectors = np.linalg.svd(n_positions - center)
        plane_axis_1, plane_axis_2 = right_vectors[0], right_vectors[1]
        polar_angles = np.arctan2(
            (n_positions - center) @ plane_axis_2,
            (n_positions - center) @ plane_axis_1,
        )
        ordered_n = n_positions[np.argsort(polar_angles)]
        dihedral = dihedral_degrees(*ordered_n)
        if np.isfinite(dihedral):
            n4_dihedrals.append(abs(dihedral))

    if analyzed_pd == 0:
        return None

    return {
        "pd_n_distances": [float(value) for value in distances],
        "cis_angles": [float(value) for value in cis_angles],
        "trans_angles": [float(value) for value in trans_angles],
        "n4_dihedrals": [float(value) for value in n4_dihedrals],
        "mean_pd_n_distance_angstrom": float(np.mean(distances)),
        "mean_cis_angle_deg": float(np.mean(cis_angles)),
        "mean_trans_angle_deg": float(np.mean(trans_angles)),
        "mean_trans_deviation_deg": float(
            np.mean(np.abs(np.asarray(trans_angles) - 180.0))
        ),
        "square_planar_angle_rmsd_deg": float(
            np.sqrt(np.mean(np.square(angular_residuals)))
        ),
        "pd_centers_analyzed": analyzed_pd,
        "num_atoms": len(atoms),
    }


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.linewidth": 0.8,
            "legend.fontsize": 10.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "savefig.dpi": 300,
        }
    )


def style_axis(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)


def draw_model_boxplot(axis, grouped_values: dict[str, list[float]], ylabel: str) -> list[str]:
    """Draw a model-colored Tukey boxplot without raw point clouds."""
    models = ordered_models(
        model for model, values in grouped_values.items() if len(values) > 0
    )
    values = [np.asarray(grouped_values[model], dtype=float) for model in models]
    boxplot = axis.boxplot(
        values,
        tick_labels=[model_label(model) for model in models],
        patch_artist=True,
        widths=0.58,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 4.5,
        },
        medianprops={"color": "black", "linewidth": 1.4},
        whiskerprops={"color": "#333333", "linewidth": 1.0},
        capprops={"color": "#333333", "linewidth": 1.0},
        flierprops={
            "marker": "o",
            "markerfacecolor": "none",
            "markeredgecolor": "#555555",
            "markersize": 3.0,
            "alpha": 0.7,
        },
    )
    for patch, model in zip(boxplot["boxes"], models):
        patch.set_facecolor(model_color(model))
        patch.set_edgecolor("#333333")
        patch.set_alpha(0.82)

    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=20)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    style_axis(axis)
    return models


def draw_density_distributions(
    axis,
    grouped_values: dict[str, list[float]],
    xlabel: str,
    *,
    expected_value: float | None = None,
    expected_label: str | None = None,
    lower_bound: float | None = None,
) -> list[str]:
    """Draw normalized Gaussian kernel-density estimates for each model."""
    models = ordered_models(
        model for model, values in grouped_values.items() if len(values) > 0
    )
    all_values = np.concatenate(
        [np.asarray(grouped_values[model], dtype=float) for model in models]
    )
    span = float(np.ptp(all_values))
    padding = 0.08 * span if span > 1.0e-10 else max(abs(all_values[0]) * 0.05, 0.1)
    x_min = float(np.min(all_values) - padding)
    x_max = float(np.max(all_values) + padding)
    if lower_bound is not None:
        x_min = max(lower_bound, x_min)
    x_grid = np.linspace(x_min, x_max, 500)

    for model in models:
        values = np.asarray(grouped_values[model], dtype=float)
        if len(values) >= 2 and float(np.std(values)) > 1.0e-12:
            density = gaussian_kde(values)(x_grid)
        else:
            sigma = max(0.02 * max(abs(float(values[0])), 1.0), 0.02)
            density = np.exp(-0.5 * ((x_grid - values[0]) / sigma) ** 2)
            density /= sigma * np.sqrt(2.0 * np.pi)
        color = model_color(model)
        axis.plot(x_grid, density, color=color, linewidth=2.0, label=model_label(model))
        axis.fill_between(x_grid, 0.0, density, color=color, alpha=0.12)

    if expected_value is not None:
        axis.axvline(
            expected_value,
            color="#4A4A4A",
            linestyle="--",
            linewidth=1.4,
            label=expected_label or "Ideal value",
        )

    axis.set_xlabel(xlabel)
    axis.set_ylabel("Probability density")
    style_axis(axis)
    return models


def group_result_values(results: dict, field: str) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for cage_name, record in results.items():
        value = record.get(field)
        if value is None:
            continue
        grouped[model_from_cage_name(cage_name)].append(float(value))
    return grouped
