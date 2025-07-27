"""
Test cases for Network Flow algorithms
"""

import pytest
from network_flow.flow_network import FlowNetwork, Edge
from network_flow.max_flow import ford_fulkerson, edmonds_karp, dinic_algorithm

class TestFlowNetwork:
    def test_flow_network_creation(self):
        """Test basic flow network creation."""
        network = FlowNetwork(4)
        
        assert network.n == 4
        assert len(network.graph) == 4
        
        # Add edges
        network.add_edge(0, 1, 10)
        network.add_edge(1, 2, 5)
        network.add_edge(2, 3, 8)
        
        # Check edges were added
        assert len(network.graph[0]) == 1
        assert network.graph[0][0].to == 1
        assert network.graph[0][0].capacity == 10
    
    def test_edge_operations(self):
        """Test edge operations."""
        edge = Edge(1, 10, 5)
        
        assert edge.residual_capacity() == 5
        assert not edge.is_saturated()
        
        edge.flow = 10
        assert edge.residual_capacity() == 0
        assert edge.is_saturated()
    
    def test_flow_network_from_matrix(self):
        """Test creating network from capacity matrix."""
        capacity_matrix = [
            [0, 10, 0, 10],
            [0, 0, 25, 6],
            [0, 0, 0, 10],
            [0, 0, 0, 0]
        ]
        
        network = FlowNetwork.from_capacity_matrix(capacity_matrix)
        
        assert network.n == 4
        # Check that edges were created correctly
        assert len(network.graph[0]) == 2  # Two outgoing edges from source

class TestMaxFlow:
    def test_ford_fulkerson_simple(self):
        """Test Ford-Fulkerson on simple network."""
        network = FlowNetwork(4)
        network.add_edge(0, 1, 10)
        network.add_edge(0, 2, 10)
        network.add_edge(1, 3, 10)
        network.add_edge(2, 3, 10)
        
        max_flow = ford_fulkerson(network, 0, 3)
        assert max_flow == 20
    
    def test_edmonds_karp_simple(self):
        """Test Edmonds-Karp on simple network."""
        network = FlowNetwork(4)
        network.add_edge(0, 1, 10)
        network.add_edge(0, 2, 10)
        network.add_edge(1, 3, 10)
        network.add_edge(2, 3, 10)
        
        max_flow = edmonds_karp(network, 0, 3)
        assert max_flow == 20
    
    def test_dinic_algorithm_simple(self):
        """Test Dinic's algorithm on simple network."""
        network = FlowNetwork(4)
        network.add_edge(0, 1, 10)
        network.add_edge(0, 2, 10)
        network.add_edge(1, 3, 10)
        network.add_edge(2, 3, 10)
        
        max_flow = dinic_algorithm(network, 0, 3)
        assert max_flow == 20
    
    def test_bottleneck_flow(self):
        """Test flow with bottleneck."""
        network = FlowNetwork(4)
        network.add_edge(0, 1, 100)
        network.add_edge(1, 2, 1)  # Bottleneck
        network.add_edge(2, 3, 100)
        
        max_flow = ford_fulkerson(network, 0, 3)
        assert max_flow == 1
    
    def test_complex_network(self):
        """Test more complex network."""
        # Create the network from CLRS textbook example
        network = FlowNetwork(6)
        network.add_edge(0, 1, 16)  # s -> v1
        network.add_edge(0, 2, 13)  # s -> v2
        network.add_edge(1, 2, 10)  # v1 -> v2
        network.add_edge(1, 3, 12)  # v1 -> v3
        network.add_edge(2, 1, 4)   # v2 -> v1
        network.add_edge(2, 4, 14)  # v2 -> v4
        network.add_edge(3, 2, 9)   # v3 -> v2
        network.add_edge(3, 5, 20)  # v3 -> t
        network.add_edge(4, 3, 7)   # v4 -> v3
        network.add_edge(4, 5, 4)   # v4 -> t
        
        # All algorithms should give same result
        max_flow_ff = ford_fulkerson(network, 0, 5)
        network.reset_flow()
        max_flow_ek = edmonds_karp(network, 0, 5)
        network.reset_flow()
        max_flow_dinic = dinic_algorithm(network, 0, 5)
        
        assert max_flow_ff == max_flow_ek == max_flow_dinic == 23
    
    def test_no_path_to_sink(self):
        """Test network with no path to sink."""
        network = FlowNetwork(4)
        network.add_edge(0, 1, 10)
        network.add_edge(2, 3, 10)
        # No path from 0 to 3
        
        max_flow = ford_fulkerson(network, 0, 3)
        assert max_flow == 0
    
    def test_disconnected_source_sink(self):
        """Test when source and sink are same."""
        network = FlowNetwork(2)
        network.add_edge(0, 1, 10)
        
        # Flow from vertex to itself should be 0
        max_flow = ford_fulkerson(network, 0, 0)
        assert max_flow == 0

class TestFlowValidation:
    def test_flow_conservation(self):
        """Test flow conservation property."""
        network = FlowNetwork(4)
        network.add_edge(0, 1, 10)
        network.add_edge(1, 2, 10)
        network.add_edge(2, 3, 10)
        
        ford_fulkerson(network, 0, 3)
        
        # Check flow conservation at intermediate vertices
        assert network.is_valid_flow(0, 3)
    
    def test_capacity_constraints(self):
        """Test that flow doesn't exceed capacity."""
        network = FlowNetwork(3)
        network.add_edge(0, 1, 5)
        network.add_edge(1, 2, 3)
        
        ford_fulkerson(network, 0, 2)
        
        # Check capacity constraints
        for vertex in range(network.n):
            for edge in network.graph[vertex]:
                assert edge.flow <= edge.capacity
