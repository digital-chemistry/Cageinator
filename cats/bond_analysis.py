#!/usr/bin/env python3
"""CATS v2: analyze local square-planar Pd(II) coordination geometry.

Input:
    ``--dir PROJECT_DIR`` containing ``assemblies/``.
Outputs:
    ``evaluation/bond_analysis/results.json`` and
    ``evaluation/bond_analysis/bond_geometry_analysis.png``.
Method:
    The four nearest N atoms within 2.5 Å of each Pd atom define the local
    coordination sphere.  The figure reports Gaussian kernel-density
    distributions for Pd--N distances, cis and trans N--Pd--N angles, and the
    absolute N4 dihedral.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

try:
    from ._shared import (
        analyze_square_planar_pd,
        build_parser,
        configure_plot_style,
        draw_density_distributions,
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
        analyze_square_planar_pd,
        build_parser,
        configure_plot_style,
        draw_density_distributions,
        load_assembly_geometry,
        model_from_cage_name,
        parse_cage_name,
        report_finish,
        report_start,
        resolve_directories,
        write_results,
    )


ANALYSIS_NAME = "bond_analysis"
FIGURE_NAME = "bond_geometry_analysis.png"


def generate_figure(results: dict, evaluation_dir: Path) -> Path:
    configure_plot_style()
    grouped = defaultdict(
        lambda: {
            "pd_n_distances": [],
            "cis_angles": [],
            "trans_angles": [],
            "n4_dihedrals": [],
        }
    )
    for cage_name, record in results.items():
        model = model_from_cage_name(cage_name)
        for field in grouped[model]:
            grouped[model][field].extend(record[field])

    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    panels = (
        (
            axes[0, 0],
            "pd_n_distances",
            r"Pd--N distance ($\mathrm{\AA}$)",
            2.05,
            r"Ideal Pd--N: 2.05 $\mathrm{\AA}$",
            0.0,
        ),
        (
            axes[0, 1],
            "cis_angles",
            r"cis N--Pd--N angle ($^\circ$)",
            90.0,
            r"Ideal cis angle: 90$^\circ$",
            0.0,
        ),
        (
            axes[1, 0],
            "trans_angles",
            r"trans N--Pd--N angle ($^\circ$)",
            180.0,
            r"Ideal trans angle: 180$^\circ$",
            0.0,
        ),
        (
            axes[1, 1],
            "n4_dihedrals",
            r"Absolute N$_4$ dihedral ($^\circ$)",
            0.0,
            r"Ideal plane: 0$^\circ$",
            0.0,
        ),
    )

    handles = labels = None
    for axis, field, xlabel, ideal, ideal_label, lower_bound in panels:
        values = {
            model: model_records[field]
            for model, model_records in grouped.items()
            if model_records[field]
        }
        draw_density_distributions(
            axis,
            values,
            xlabel,
            expected_value=ideal,
            expected_label=ideal_label,
            lower_bound=lower_bound,
        )
        if handles is None:
            handles, labels = axis.get_legend_handles_labels()

    if handles and labels:
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.015),
            ncol=5,
            frameon=False,
        )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figure_path = evaluation_dir / FIGURE_NAME
    figure.savefig(figure_path, bbox_inches="tight")
    plt.close(figure)
    return figure_path


def main() -> None:
    parser = build_parser("CATS: analyze square-planar Pd(II) geometry")
    args = parser.parse_args()
    _, assemblies_dir, evaluation_dir = resolve_directories(args.dir, ANALYSIS_NAME)
    report_start(ANALYSIS_NAME, assemblies_dir)

    results = {}
    for cage_dir in sorted(assemblies_dir.iterdir()):
        if not cage_dir.is_dir():
            continue
        atoms, source_file = load_assembly_geometry(cage_dir)
        if not atoms:
            continue
        geometry = analyze_square_planar_pd(atoms)
        metadata = parse_cage_name(cage_dir.name)
        if geometry is None or metadata is None:
            continue
        geometry["cage_name"] = cage_dir.name
        geometry.update(metadata)
        geometry["source_file"] = source_file
        results[cage_dir.name] = geometry

    results_path = write_results(results, evaluation_dir)
    figure_path = generate_figure(results, evaluation_dir) if results else None
    report_finish(
        ANALYSIS_NAME,
        len(results),
        results_path,
        figure_path,
    )


if __name__ == "__main__":
    main()
