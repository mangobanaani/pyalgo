"""
Network Flow algorithms package.

This package contains implementations of various network flow algorithms including:
- Maximum flow algorithms (Ford-Fulkerson, Edmonds-Karp, Dinic's)
- Minimum cost flow algorithms
- Maximum matching algorithms
- Flow network data structures
"""

from .flow_network import FlowNetwork, Edge
from .max_flow import ford_fulkerson, edmonds_karp, dinic_algorithm
from .min_cost_flow import (
    MinCostFlowNetwork, min_cost_max_flow, min_cost_flow, 
    cycle_canceling, transportation_problem
)
from .matching import (
    maximum_bipartite_matching, hopcroft_karp, hungarian_algorithm,
    maximum_weight_matching, stable_marriage
)

__all__ = [
    'FlowNetwork',
    'Edge',
    'ford_fulkerson',
    'edmonds_karp', 
    'dinic_algorithm',
    'MinCostFlowNetwork',
    'min_cost_max_flow',
    'min_cost_flow',
    'cycle_canceling',
    'transportation_problem',
    'maximum_bipartite_matching',
    'hopcroft_karp',
    'hungarian_algorithm',
    'maximum_weight_matching',
    'stable_marriage'
]
