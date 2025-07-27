"""
Utility functions module
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def validate_smiles(smiles: str) -> bool:
    """Validate SMILES format correctness"""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False

def calculate_molecular_properties(mol) -> Dict[str, float]:
    """Calculate basic molecular properties"""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    
    if mol is None:
        return {}
    
    return {
        'molecular_weight': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'num_atoms': mol.GetNumAtoms(),
        'num_bonds': mol.GetNumBonds(),
        'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol)
    }

class QuantumDockingError(Exception):
    """Custom exception class for quantum docking"""
    pass
