#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared figure palette and typography for paper experiment plots."""

from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt


# User-supplied palette, with invalid/mismatched RGB-to-hex samples corrected.
USER_PALETTE: Dict[str, str] = {
    "sky": "#58BCF1",
    "light_gray": "#D4D4D4",
    "blue": "#6EAFDD",
    "lavender_gray": "#B1B2C9",
    "magenta": "#D72E9E",
    "pink": "#E1ADD5",
    "navy": "#2C2C64",
    "salmon": "#F79173",
    "violet": "#B182C9",
}


METHOD_COLORS_BY_KEY: Dict[str, str] = {
    "market_random_select": USER_PALETTE["light_gray"],
    "market_coreset_select": USER_PALETTE["sky"],
    "market_cosine_select": USER_PALETTE["blue"],
    "market_badge_select": USER_PALETTE["pink"],
    "market_kmeans_center_select": USER_PALETTE["salmon"],
    "market_typiclust_select": USER_PALETTE["lavender_gray"],
    "market_uncertainty_select": USER_PALETTE["violet"],
    "ours_kernel_ridge_student": USER_PALETTE["magenta"],
    "ours_sample_package_krr": USER_PALETTE["navy"],
}


METHOD_COLORS_BY_LABEL: Dict[str, str] = {
    "Random": USER_PALETTE["light_gray"],
    "CoreSet": USER_PALETTE["sky"],
    "Cosine": USER_PALETTE["blue"],
    "BADGE": USER_PALETTE["pink"],
    "KMeans": USER_PALETTE["salmon"],
    "KMeans-center": USER_PALETTE["salmon"],
    "TypiClust": USER_PALETTE["lavender_gray"],
    "Uncertainty": USER_PALETTE["violet"],
    "Raw KRR": USER_PALETTE["magenta"],
    "Pkg KRR": USER_PALETTE["navy"],
}


DATASET_COLORS: Dict[str, str] = {
    "hateful_memes": USER_PALETTE["sky"],
    "hatespeech": USER_PALETTE["magenta"],
    "mscoco": USER_PALETTE["salmon"],
    "overall": USER_PALETTE["navy"],
}


def apply_paper_style(base_size: float = 10.0) -> None:
    """Use LaTeX-rendered serif text and embeddable PDF fonts."""
    plt.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{mathpazo}",
            "font.family": "serif",
            "font.serif": ["Palatino", "Palatino Linotype", "URW Palladio L", "DejaVu Serif"],
            "font.size": base_size,
            "axes.titlesize": base_size,
            "axes.labelsize": base_size,
            "xtick.labelsize": base_size,
            "ytick.labelsize": base_size,
            "legend.fontsize": base_size,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
