"""
Integration test script
"""

from src.quantum_docking import MolecularParser, QuantumOptimizer
import numpy as np

def main():
    print("Starting integration tests...")
    
    # Test molecular parsing
    parser = MolecularParser()
    
    # Parse simple molecules
    mol1 = parser.parse_smiles("CCO", mol_id="ethanol")
    mol2 = parser.parse_smiles("CCN", mol_id="ethylamine")
    
    print(f"✓ Successfully parsed molecules: {mol1.GetNumAtoms()} and {mol2.GetNumAtoms()} atoms")
    
    # Generate conformers
    conformers1 = parser.generate_conformers(mol1, num_conformers=3)
    conformers2 = parser.generate_conformers(mol2, num_conformers=3)
    
    print(f"✓ Generated conformers: {len(conformers1)} and {len(conformers2)}")
    
    # Calculate interaction matrix
    interaction_matrix = parser.calculate_interaction_matrix(mol1, mol2)
    print(f"✓ Calculated interaction matrix: {interaction_matrix.shape}")
    
    # Quantum optimization
    optimizer = QuantumOptimizer(backend="pennylane", num_qubits=4)
    
    result = optimizer.optimize_docking_pose(
        interaction_matrix, 
        max_iterations=20
    )
    
    print(f"✓ Quantum optimization completed: Final energy = {result['optimal_energy']:.6f}")
    
    # Evaluate binding affinity
    affinity = optimizer.evaluate_binding_affinity(result['optimal_params'])
    print(f"✓ Binding affinity: {affinity:.6f}")
    
    print("\nAll integration tests passed!")

if __name__ == "__main__":
    main()
