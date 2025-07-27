"""
Docking engine tests
"""

import pytest
import numpy as np
from src.quantum_docking import DockingEngine, MolecularParser, QuantumOptimizer

class TestDockingEngine:
    
    def setup_method(self):
        """Setup before tests"""
        self.engine = DockingEngine()
    
    def test_initialization(self):
        """Test docking engine initialization"""
        assert isinstance(self.engine.parser, MolecularParser)
        assert isinstance(self.engine.optimizer, QuantumOptimizer)
        assert self.engine.scoring_function == "quantum_enhanced"
    
    def test_single_docking(self):
        """Test single molecule docking"""
        ligand = "CCO"  # Ethanol
        receptor = "CCN"  # Ethylamine
        
        result = self.engine.dock(ligand, receptor, num_poses=3)
        
        assert 'binding_affinity' in result
        assert 'optimization_result' in result
        assert 'docking_time' in result
        assert isinstance(result['binding_affinity'], float)
    
    def test_batch_docking(self):
        """Test batch docking"""
        ligands = ["CCO", "CCN", "c1ccccc1"]
        receptor = "CC(=O)O"
        
        results = self.engine.batch_dock(ligands, receptor, num_poses=2)
        
        assert len(results) == len(ligands)
        assert all('binding_affinity' in r or 'error' in r for r in results)
    
    def test_pose_ranking(self):
        """Test pose ranking functionality"""
        # Create mock results
        mock_results = [
            {'binding_affinity': -3.0},
            {'binding_affinity': -5.0},
            {'binding_affinity': -1.0}
        ]
        
        ranked = self.engine.rank_poses(mock_results)
        
        # Should be sorted by binding affinity (lowest first)
        assert ranked[0]['binding_affinity'] == -5.0
        assert ranked[1]['binding_affinity'] == -3.0
        assert ranked[2]['binding_affinity'] == -1.0
    
    def test_best_pose(self):
        """Test best pose selection"""
        mock_results = [
            {'binding_affinity': -3.0},
            {'binding_affinity': -5.0},
            {'binding_affinity': -1.0}
        ]
        
        best = self.engine.get_best_pose(mock_results)
        
        assert best['binding_affinity'] == -5.0
    
    def test_report_generation(self):
        """Test report generation"""
        # Run a simple docking first
        self.engine.dock("CCO", "CCN", num_poses=1)
        
        report = self.engine.generate_report()
        
        assert "QUANTUM MOLECULAR DOCKING REPORT" in report
        assert "BEST POSE:" in report
        assert "SUMMARY STATISTICS:" in report
