"""
Molecular parsing module - Handles SMILES format and molecular structures
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdMolAlign import AlignMol
import logging

from .utils import setup_logging, QuantumDockingError

logger = setup_logging()

class MolecularParser:
    """SMILES molecular parser"""
    
    def __init__(self):
        """Initialize molecular parser"""
        self.molecules = {}
        logger.info("Molecular parser initialized successfully")
    
    def parse_smiles(self, smiles: str, mol_id: Optional[str] = None) -> Chem.Mol:
        """
        Parse SMILES string to RDKit molecule object
        
        Args:
            smiles: SMILES string
            mol_id: Molecule identifier
            
        Returns:
            RDKit molecule object
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise QuantumDockingError(f"Cannot parse SMILES: {smiles}")
            
            # Add hydrogen atoms
            mol = Chem.AddHs(mol)
            
            # Store molecule
            if mol_id:
                self.molecules[mol_id] = mol
                
            logger.info(f"Successfully parsed SMILES: {smiles}")
            return mol
            
        except Exception as e:
            logger.error(f"SMILES parsing failed: {e}")
            raise QuantumDockingError(f"SMILES parsing error: {e}")
    
    def generate_conformers(self, mol: Chem.Mol, num_conformers: int = 10) -> List[int]:
        """
        Generate molecular conformers
        
        Args:
            mol: RDKit molecule object
            num_conformers: Number of conformers to generate
            
        Returns:
            List of conformer IDs
        """
        try:
            # Generate 3D conformers
            conformer_ids = AllChem.EmbedMultipleConfs(
                mol, 
                numConfs=num_conformers,
                randomSeed=42,
                useExpTorsionAnglePrefs=True,
                useBasicKnowledge=True
            )
            
            # Optimize conformers
            for conf_id in conformer_ids:
                AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
            
            logger.info(f"Generated {len(conformer_ids)} conformers")
            return list(conformer_ids)
            
        except Exception as e:
            logger.error(f"Conformer generation failed: {e}")
            raise QuantumDockingError(f"Conformer generation error: {e}")
    
    def get_molecular_features(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Extract molecular features
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary of molecular features
        """
        features = {
            'molecular_weight': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            'num_hba': Descriptors.NumHBA(mol),
            'num_hbd': Descriptors.NumHBD(mol),
            'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
            'num_saturated_rings': Descriptors.NumSaturatedRings(mol),
        }
        
        logger.debug(f"Extracted molecular features: {features}")
        return features
    
    def get_atom_coordinates(self, mol: Chem.Mol, conf_id: int = 0) -> np.ndarray:
        """
        Get atomic coordinates
        
        Args:
            mol: RDKit molecule object
            conf_id: Conformer ID
            
        Returns:
            Atomic coordinates array (N, 3)
        """
        try:
            conformer = mol.GetConformer(conf_id)
            coords = []
            
            for i in range(mol.GetNumAtoms()):
                pos = conformer.GetAtomPosition(i)
                coords.append([pos.x, pos.y, pos.z])
            
            return np.array(coords)
            
        except Exception as e:
            logger.error(f"Failed to get coordinates: {e}")
            raise QuantumDockingError(f"Coordinate extraction error: {e}")
    
    def calculate_interaction_matrix(self, mol1: Chem.Mol, mol2: Chem.Mol) -> np.ndarray:
        """
        Calculate interaction matrix between two molecules
        
        Args:
            mol1: First molecule
            mol2: Second molecule
            
        Returns:
            Interaction matrix
        """
        coords1 = self.get_atom_coordinates(mol1)
        coords2 = self.get_atom_coordinates(mol2)
        
        # Calculate distance matrix
        distances = np.linalg.norm(
            coords1[:, np.newaxis] - coords2[np.newaxis, :], 
            axis=2
        )
        
        # Simple Lennard-Jones potential approximation
        interaction_matrix = 4 * ((1/distances)**12 - (1/distances)**6)
        
        # Avoid infinite values
        interaction_matrix = np.clip(interaction_matrix, -100, 100)
        
        return interaction_matrix
