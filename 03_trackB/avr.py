"""
avr.py — Artery-Vein Ratio (AVR) from graph edge attributes.
"""

import networkx as nx
import numpy as np


def compute_avr(graph: nx.Graph) -> float:
    """
    Calculate Artery-Vein Ratio from mean thickness of artery vs vein edges.

    AVR = mean_artery_thickness / mean_vein_thickness

    Uses `avg_thickness` and `vessel_type` edge attributes set during
    graph extraction.

    Args:
        graph: networkx graph with edge attrs `avg_thickness` and `vessel_type`.

    Returns:
        AVR value. Returns 0.0 if either artery or vein edges are absent.
    """
    artery_thick = []
    vein_thick = []

    for _, _, data in graph.edges(data=True):
        vtype = data.get("vessel_type", "unknown")
        thickness = data.get("avg_thickness", 0.0)

        if vtype == "artery":
            artery_thick.append(thickness)
        elif vtype == "vein":
            vein_thick.append(thickness)

    if not artery_thick or not vein_thick:
        return 0.0

    mean_artery = np.mean(artery_thick)
    mean_vein = np.mean(vein_thick)

    if mean_vein == 0:
        return 0.0

    return mean_artery / mean_vein
