"""
Parameter sweep experiments for optimization
"""

import numpy as np
from typing import List, Dict, Tuple
from ..quantum_docking import QuantumOptimizer

class ParameterSweep:
    """Parameter sweep for quantum optimization"""
    
    def __init__(self):
        self.sweep_results = []
    
    def sweep_qubits(self, 
                    interaction_matrix: np.ndarray,
                    qubit_range: List[int] = [2, 4, 6, 8]) -> Dict:
        """
        Sweep over different numbers of qubits
        
        Args:
            interaction_matrix: Test interaction matrix
            qubit_range: Range of qubit numbers to test
            
        Returns:
            Sweep results
        """
        results = {}
        
        for num_qubits in qubit_range:
            optimizer = QuantumOptimizer(num_qubits=num_qubits)
            result = optimizer.optimize_docking_pose(interaction_matrix, max_iterations=30)
            
            results[num_qubits] = {
                'optimal_energy': result['optimal_energy'],
                'num_iterations': result['num_iterations']
            }
        
        return results
