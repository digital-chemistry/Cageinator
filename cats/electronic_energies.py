#!/usr/bin/env python3
"""CATS v2: compare relative GFN-FF energies across assembly models.

Input:
    ``--dir PROJECT_DIR`` containing ``assemblies/``.
Outputs:
    ``evaluation/electronic_energies/results.json`` and
    ``evaluation/electronic_energies/energy_preferences.png``.
Method:
    The energy in each optimized XYZ header is normalized by the number of
    [ML2] units.  Relative energies are evaluated within each node--linker
    family.  The figure combines energy boxplots, a correlation with trans
    N--Pd--N angular distortion, and lowest-energy model frequencies.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from ._shared import (
        analyze_square_planar_pd,
        build_parser,
        configure_plot_style,
        draw_model_boxplot,
        load_assembly_geometry,
        model_color,
        model_from_cage_name,
        model_label,
        ordered_models,
        parse_cage_name,
        report_finish,
        report_start,
        resolve_directories,
        style_axis,
        write_results,
    )
except ImportError:
    from _shared import (
        analyze_square_planar_pd,
        build_parser,
        configure_plot_style,
        draw_model_boxplot,
        load_assembly_geometry,
        model_color,
        model_from_cage_name,
        model_label,
        ordered_models,
        parse_cage_name,
        report_finish,
        report_start,
        resolve_directories,
        style_axis,
        write_results,
    )


ANALYSIS_NAME = "electronic_energies"
FIGURE_NAME = "energy_preferences.png"
HARTREE_TO_KCAL_MOL = 627.509
ENERGY_PATTERN = re.compile(
    r"energy:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?)",
    re.IGNORECASE,
)


def extract_xyz_energy(path: Path) -> tuple[float | None, int]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            atom_count = int(handle.readline().strip())
            comment = handle.readline()
    except (OSError, ValueError):
        return None, 0

    match = ENERGY_PATTERN.search(comment)
    if match is None:
        return None, atom_count
    return float(match.group(1).replace("D", "E").replace("d", "e")), atom_count


def generate_figure(results: dict, evaluation_dir: Path) -> Path:
    configure_plot_style()
    figure, (energy_axis, correlation_axis, preference_axis) = plt.subplots(
        1,
        3,
        figsize=(14, 5.5),
        gridspec_kw={"width_ratios": [1.25, 1.35, 0.9]},
    )

    grouped_energies = defaultdict(list)
    for record in results.values():
        grouped_energies[record["model_key"]].append(
            record["relative_energy_per_ml2_unit_kcal_mol"]
        )
    draw_model_boxplot(
        energy_axis,
        grouped_energies,
        r"$\Delta E$ per [ML$_2$] unit (kcal mol$^{-1}$)",
    )
    energy_axis.axhline(0.0, color="#555555", linewidth=0.9, zorder=0)

    correlation_records = [
        record
        for record in results.values()
        if record.get("mean_trans_deviation_deg") is not None
    ]
    models = ordered_models(record["model_key"] for record in correlation_records)
    for model in models:
        model_records = [
            record for record in correlation_records if record["model_key"] == model
        ]
        correlation_axis.scatter(
            [record["mean_trans_deviation_deg"] for record in model_records],
            [
                record["relative_energy_per_ml2_unit_kcal_mol"]
                for record in model_records
            ],
            color=model_color(model),
            edgecolor="white",
            linewidth=0.4,
            alpha=0.78,
            s=34,
            label=model_label(model),
            zorder=3,
        )

    preferred = [record for record in correlation_records if record["is_preferred"]]
    correlation_axis.scatter(
        [record["mean_trans_deviation_deg"] for record in preferred],
        [record["relative_energy_per_ml2_unit_kcal_mol"] for record in preferred],
        marker="*",
        facecolor="white",
        edgecolor="black",
        linewidth=0.8,
        s=78,
        label="Lowest energy in family",
        zorder=5,
    )

    x_values = np.asarray(
        [record["mean_trans_deviation_deg"] for record in correlation_records],
        dtype=float,
    )
    y_values = np.asarray(
        [
            record["relative_energy_per_ml2_unit_kcal_mol"]
            for record in correlation_records
        ],
        dtype=float,
    )
    if len(x_values) >= 2 and np.std(x_values) > 0.0 and np.std(y_values) > 0.0:
        slope, intercept = np.polyfit(x_values, y_values, 1)
        x_line = np.linspace(float(np.min(x_values)), float(np.max(x_values)), 400)
        fitted_energy = slope * x_line + intercept

        trend_floor = max(0.5, 0.01 * max(float(np.ptp(y_values)), 1.0))
        visible_fit = fitted_energy >= trend_floor
        if np.count_nonzero(visible_fit) >= 2:
            correlation_axis.plot(
                x_line[visible_fit],
                fitted_energy[visible_fit],
                color="#4A4A4A",
                linestyle="--",
                linewidth=1.3,
                zorder=2,
            )

        pearson_r = float(np.corrcoef(x_values, y_values)[0, 1])
        correlation_axis.annotate(
            rf"Pearson $r={pearson_r:.2f}$",
            xy=(1.0, 1.0),
            xycoords="axes fraction",
            xytext=(0, 7),
            textcoords="offset points",
            horizontalalignment="right",
            verticalalignment="bottom",
            annotation_clip=False,
        )

    correlation_axis.set_xlabel(
        r"Mean trans-angle deviation, $|180^\circ-\theta_\mathrm{trans}|$ ($^\circ$)"
    )
    correlation_axis.set_ylabel(
        r"$\Delta E$ per [ML$_2$] unit (kcal mol$^{-1}$)"
    )
    style_axis(correlation_axis)
    correlation_axis.margins(x=0.04, y=0.06)
    legend_handles, legend_labels = correlation_axis.get_legend_handles_labels()

    preferred_counts = defaultdict(int)
    for record in results.values():
        if record["is_preferred"]:
            preferred_counts[record["model_key"]] += 1
    system_count = len({record["system_key"] for record in results.values()})
    preference_models = ordered_models(record["model_key"] for record in results.values())
    percentages = [
        100.0 * preferred_counts[model] / system_count if system_count else 0.0
        for model in preference_models
    ]
    bars = preference_axis.bar(
        range(len(preference_models)),
        percentages,
        color=[model_color(model) for model in preference_models],
        edgecolor="#333333",
        linewidth=0.8,
        width=0.62,
        zorder=3,
    )
    preference_axis.set_xticks(
        range(len(preference_models)),
        [model_label(model) for model in preference_models],
        rotation=20,
        horizontalalignment="right",
    )
    preference_axis.set_ylabel("Lowest-energy frequency (%)")
    preference_axis.set_ylim(0.0, max(65.0, max(percentages, default=0.0) + 8.0))
    style_axis(preference_axis)
    for bar, percentage in zip(bars, percentages):
        preference_axis.annotate(
            f"{percentage:.1f}%",
            (bar.get_x() + bar.get_width() / 2.0, percentage),
            xytext=(0, 3),
            textcoords="offset points",
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=9,
        )

    figure.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=5,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    figure_path = evaluation_dir / FIGURE_NAME
    figure.savefig(figure_path, bbox_inches="tight")
    plt.close(figure)
    return figure_path


def main() -> None:
    parser = build_parser("CATS: compare relative GFN-FF assembly energies")
    args = parser.parse_args()
    _, assemblies_dir, evaluation_dir = resolve_directories(args.dir, ANALYSIS_NAME)
    report_start(ANALYSIS_NAME, assemblies_dir)

    family_records = defaultdict(list)
    for cage_dir in sorted(assemblies_dir.iterdir()):
        if not cage_dir.is_dir():
            continue

        optimized_xyz = cage_dir / "xtb_opt.xyz"
        if not optimized_xyz.is_file() or optimized_xyz.stat().st_size == 0:
            continue
        energy_hartree, atom_count = extract_xyz_energy(optimized_xyz)
        metadata = parse_cage_name(cage_dir.name)
        if energy_hartree is None or metadata is None:
            continue
        if metadata["linker_count"] != 2 * metadata["metal_count"]:
            continue

        atoms, source_file = load_assembly_geometry(
            cage_dir,
            require_optimized=True,
        )
        local_geometry = analyze_square_planar_pd(atoms)
        record = {
            "cage_name": cage_dir.name,
            **metadata,
            "source_file": source_file,
            "atom_count": atom_count,
            "total_energy_hartree": energy_hartree,
            "energy_per_ml2_unit_hartree": energy_hartree
            / metadata["ml2_units"],
            "mean_trans_angle_deg": (
                local_geometry["mean_trans_angle_deg"]
                if local_geometry is not None
                else None
            ),
            "mean_trans_deviation_deg": (
                local_geometry["mean_trans_deviation_deg"]
                if local_geometry is not None
                else None
            ),
            "square_planar_angle_rmsd_deg": (
                local_geometry["square_planar_angle_rmsd_deg"]
                if local_geometry is not None
                else None
            ),
        }
        family_records[metadata["system_key"]].append(record)

    results = {}
    for records in family_records.values():
        family_minimum = min(
            record["energy_per_ml2_unit_hartree"] for record in records
        )
        for record in records:
            delta_hartree = record["energy_per_ml2_unit_hartree"] - family_minimum
            record["relative_energy_per_ml2_unit_kcal_mol"] = (
                delta_hartree * HARTREE_TO_KCAL_MOL
            )
            record["is_preferred"] = bool(
                np.isclose(delta_hartree, 0.0, rtol=0.0, atol=1.0e-12)
            )
            results[record["cage_name"]] = record

    results_path = write_results(results, evaluation_dir)
    figure_path = generate_figure(results, evaluation_dir) if results else None
    report_finish(ANALYSIS_NAME, len(results), results_path, figure_path)


if __name__ == "__main__":
    main()
