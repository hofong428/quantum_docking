"""
Quantum optimizer tests
"""

import pytest
import numpy as np

from src.quantum_docking.quantum_optimizer import QuantumOptimizer

class TestQuantumOptimizer:
    
    def setup_method(self):
        """Setup before tests"""
        self.optimizer = QuantumOptimizer(backend="pennylane", num_qubits=4)
    
    def test_initialization(self):
        """Test initialization"""
        assert self.optimizer.backend == "pennylane"
        assert self.optimizer.num_qubits == 4
        assert hasattr(self.optimizer, 'device')
    
    def test_encode_molecular_problem(self):
        """Test molecular problem encoding"""
        # Create simulated interaction matrix
        interaction_matrix = np.random.random((10, 10))
        
        problem = self.optimizer.encode_molecular_problem(interaction_matrix)
        
        assert 'hamiltonian_coeffs' in problem
        assert 'num_params' in problem
        assert len(problem['hamiltonian_coeffs']) == self.optimizer.num_qubits
    
    def test_optimize_docking_pose(self):
        """Test docking pose optimization"""
        # Create simple interaction matrix
        interaction_matrix = np.eye(10) * 0.1
        
        result = self.optimizer.optimize_docking_pose(
            interaction_matrix, 
            max_iterations=10
        )
        
        required_keys = [
            'optimal_params', 'optimal_energy', 
            'optimization_history', 'num_iterations'
        ]
        
        for key in required_keys:
            assert key in result
    
    def test_evaluate_binding_affinity(self):
        """Test binding affinity evaluation"""
        params = np.random.uniform(0, 2*np.pi, 8)
        
        affinity = self.optimizer.evaluate_binding_affinity(params)
        
        assert isinstance(affinity, float)
        assert affinity <= 0  # Binding affinity should be negative
