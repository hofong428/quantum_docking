"""
Docking engine module - Main orchestrator for molecular docking workflows
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from rdkit import Chem
import logging
import time

from .molecular_parser import MolecularParser
from .quantum_optimizer import QuantumOptimizer
from .utils import setup_logging, QuantumDockingError

logger = setup_logging()

class DockingEngine:
    """Main docking engine that orchestrates the entire docking workflow"""
    
    def __init__(self, 
                 molecular_parser: Optional[MolecularParser] = None,
                 quantum_optimizer: Optional[QuantumOptimizer] = None,
                 scoring_function: str = "quantum_enhanced"):
        """
        Initialize docking engine
        
        Args:
            molecular_parser: Molecular parser instance
            quantum_optimizer: Quantum optimizer instance
            scoring_function: Type of scoring function to use
        """
        self.parser = molecular_parser or MolecularParser()
        self.optimizer = quantum_optimizer or QuantumOptimizer()
        self.scoring_function = scoring_function
        self.docking_results = []
        
        logger.info(f"Docking engine initialized with {scoring_function} scoring")
    
    def dock(self, 
             ligand: Union[str, Chem.Mol], 
             receptor: Union[str, Chem.Mol],
             num_poses: int = 10,
             energy_threshold: float = -5.0) -> Dict:
        """
        Perform molecular docking
        
        Args:
            ligand: Ligand molecule (SMILES string or RDKit Mol)
            receptor: Receptor molecule (SMILES string or RDKit Mol)
            num_poses: Number of poses to generate
            energy_threshold: Energy threshold for pose filtering
            
        Returns:
            Docking results dictionary
        """
        start_time = time.time()
        
        try:
            # Parse molecules if they are SMILES strings
            if isinstance(ligand, str):
                ligand_mol = self.parser.parse_smiles(ligand, mol_id="ligand")
            else:
                ligand_mol = ligand
                
            if isinstance(receptor, str):
                receptor_mol = self.parser.parse_smiles(receptor, mol_id="receptor")
            else:
                receptor_mol = receptor
            
            logger.info(f"Starting docking: Ligand({ligand_mol.GetNumAtoms()} atoms) vs Receptor({receptor_mol.GetNumAtoms()} atoms)")
            
            # Generate conformers for ligand
            ligand_conformers = self.parser.generate_conformers(ligand_mol, num_conformers=num_poses)
            receptor_conformers = self.parser.generate_conformers(receptor_mol, num_conformers=1)
            
            # Calculate interaction matrix
            interaction_matrix = self.parser.calculate_interaction_matrix(ligand_mol, receptor_mol)
            
            # Quantum optimization
            optimization_result = self.optimizer.optimize_docking_pose(
                interaction_matrix, 
                max_iterations=50
            )
            
            # Calculate binding affinity
            binding_affinity = self.optimizer.evaluate_binding_affinity(
                optimization_result['optimal_params']
            )
            
            # Extract molecular features
            ligand_features = self.parser.get_molecular_features(ligand_mol)
            receptor_features = self.parser.get_molecular_features(receptor_mol)
            
            # Compile results
            docking_result = {
                'ligand_features': ligand_features,
                'receptor_features': receptor_features,
                'binding_affinity': binding_affinity,
                'optimization_result': optimization_result,
                'num_conformers': len(ligand_conformers),
                'interaction_matrix_shape': interaction_matrix.shape,
                'docking_time': time.time() - start_time,
                'energy_threshold': energy_threshold,
                'passes_threshold': binding_affinity < energy_threshold
            }
            
            self.docking_results.append(docking_result)
            
            logger.info(f"Docking completed in {docking_result['docking_time']:.2f}s")
            logger.info(f"Binding affinity: {binding_affinity:.6f}")
            
            return docking_result
            
        except Exception as e:
            logger.error(f"Docking failed: {e}")
            raise QuantumDockingError(f"Docking error: {e}")
    
    def batch_dock(self, 
                   ligand_list: List[str], 
                   receptor: Union[str, Chem.Mol],
                   **kwargs) -> List[Dict]:
        """
        Perform batch docking for multiple ligands
        
        Args:
            ligand_list: List of ligand SMILES strings
            receptor: Receptor molecule
            **kwargs: Additional arguments for dock method
            
        Returns:
            List of docking results
        """
        logger.info(f"Starting batch docking for {len(ligand_list)} ligands")
        
        batch_results = []
        
        for i, ligand in enumerate(ligand_list):
            try:
                logger.info(f"Processing ligand {i+1}/{len(ligand_list)}: {ligand}")
                result = self.dock(ligand, receptor, **kwargs)
                batch_results.append(result)
                
            except Exception as e:
                logger.warning(f"Failed to dock ligand {ligand}: {e}")
                batch_results.append({'error': str(e), 'ligand': ligand})
        
        logger.info(f"Batch docking completed: {len(batch_results)} results")
        return batch_results
    
    def rank_poses(self, docking_results: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Rank docking poses by binding affinity
        
        Args:
            docking_results: List of docking results (uses self.docking_results if None)
            
        Returns:
            Sorted list of docking results
        """
        results = docking_results or self.docking_results
        
        # Sort by binding affinity (lower is better)
        ranked_results = sorted(
            results, 
            key=lambda x: x.get('binding_affinity', float('inf'))
        )
        
        logger.info(f"Ranked {len(ranked_results)} poses")
        return ranked_results
    
    def get_best_pose(self, docking_results: Optional[List[Dict]] = None) -> Optional[Dict]:
        """
        Get the best docking pose
        
        Args:
            docking_results: List of docking results
            
        Returns:
            Best docking result or None
        """
        ranked_poses = self.rank_poses(docking_results)
        
        if ranked_poses:
            best_pose = ranked_poses[0]
            logger.info(f"Best pose binding affinity: {best_pose['binding_affinity']:.6f}")
            return best_pose
        
        return None
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate docking report
        
        Args:
            output_file: Output file path (optional)
            
        Returns:
            Report string
        """
        if not self.docking_results:
            return "No docking results available."
        
        report_lines = [
            "=" * 60,
            "QUANTUM MOLECULAR DOCKING REPORT",
            "=" * 60,
            f"Total docking runs: {len(self.docking_results)}",
            f"Scoring function: {self.scoring_function}",
            ""
        ]
        
        # Best pose information
        best_pose = self.get_best_pose()
        if best_pose:
            report_lines.extend([
                "BEST POSE:",
                f"  Binding Affinity: {best_pose['binding_affinity']:.6f}",
                f"  Docking Time: {best_pose['docking_time']:.2f}s",
                f"  Optimization Iterations: {best_pose['optimization_result']['num_iterations']}",
                ""
            ])
        
        # Summary statistics
        affinities = [r['binding_affinity'] for r in self.docking_results if 'binding_affinity' in r]
        if affinities:
            report_lines.extend([
                "SUMMARY STATISTICS:",
                f"  Mean Binding Affinity: {np.mean(affinities):.6f}",
                f"  Std Binding Affinity: {np.std(affinities):.6f}",
                f"  Min Binding Affinity: {np.min(affinities):.6f}",
                f"  Max Binding Affinity: {np.max(affinities):.6f}",
                ""
            ])
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_file}")
        
        return report
