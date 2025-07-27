# Quantum Docking API Documentation

## Core Classes

### MolecularParser

Main class for handling molecular structures and SMILES parsing.

#### Methods

##### `parse_smiles(smiles: str, mol_id: Optional[str] = None) -> Chem.Mol`
Parse SMILES string to RDKit molecule object.

**Parameters:**
- `smiles`: SMILES string representation
- `mol_id`: Optional molecule identifier

**Returns:** RDKit Mol object

**Example:**
```python
parser = MolecularParser()
mol = parser.parse_smiles("CCO", mol_id="ethanol")
