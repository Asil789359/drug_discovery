from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import numpy as np

class MolecularFeatureExtractor:
    def __init__(self):
        pass

    def smiles_to_mol(self, smiles):
        """Convert SMILES string to RDKit molecule object"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            return mol
        except Exception as e:
            raise ValueError(f"Error parsing SMILES {smiles}: {str(e)}")

    def compute_descriptors(self, mol):
        """Compute molecular descriptors"""
        descriptors = {
            'MolWt': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'HBD': Descriptors.NumHDonors(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'RotatableBonds': Descriptors.NumRotatableBonds(mol),
            'HeavyAtoms': Descriptors.HeavyAtomCount(mol)
        }
        return descriptors

    def generate_fingerprint(self, mol, fp_type='morgan', radius=2, nBits=2048):
        """Generate molecular fingerprint"""
        if fp_type == 'morgan':
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        else:
            fp = Chem.RDKFingerprint(mol, fpSize=nBits)

        return np.array(fp)

    def extract_features(self, smiles):
        """Extract all features from SMILES"""
        mol = self.smiles_to_mol(smiles)
        descriptors = self.compute_descriptors(mol)
        fingerprint = self.generate_fingerprint(mol)

        features = {
            'descriptors': descriptors,
            'fingerprint': fingerprint,
            'smiles': smiles
        }

        return features
