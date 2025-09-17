"""Chemical feature extraction and graph representations."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import torch_geometric
from torch_geometric.data import Data

from moa.utils.config import Config
from moa.utils.logger import get_logger

logger = get_logger(__name__)


class MolecularGraphFeaturizer:
    """Convert molecules to graph representations with rich node and edge features."""
    
    def __init__(self, config: Config):
        """
        Initialize molecular graph featurizer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.node_features = config.get("features.chemistry.graph_features.node_features", [])
        self.edge_features = config.get("features.chemistry.graph_features.edge_features", [])
        
        # Atom feature mappings
        self.atom_features = {
            'atomic_number': lambda atom: atom.GetAtomicNum(),
            'formal_charge': lambda atom: atom.GetFormalCharge(),
            'hybridization': lambda atom: atom.GetHybridization().real,
            'aromatic': lambda atom: int(atom.GetIsAromatic()),
            'degree': lambda atom: atom.GetDegree(),
            'total_valence': lambda atom: atom.GetTotalValence(),
            'implicit_valence': lambda atom: atom.GetImplicitValence(),
            'radical_electrons': lambda atom: atom.GetNumRadicalElectrons(),
            'in_ring': lambda atom: int(atom.IsInRing()),
            'chiral_tag': lambda atom: atom.GetChiralTag().real
        }
        
        # Bond feature mappings
        self.bond_features = {
            'bond_type': lambda bond: bond.GetBondType().real,
            'conjugated': lambda bond: int(bond.GetIsConjugated()),
            'in_ring': lambda bond: int(bond.IsInRing()),
            'stereo': lambda bond: bond.GetStereo().real,
            'aromatic': lambda bond: int(bond.GetIsAromatic())
        }
    
    def smiles_to_graph(self, smiles: str) -> Optional[Data]:
        """
        Convert SMILES to PyTorch Geometric graph.
        
        Args:
            smiles: SMILES string
            
        Returns:
            PyTorch Geometric Data object or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Add hydrogens for complete graph
            mol = Chem.AddHs(mol)
            
            # Extract node features
            node_features = []
            for atom in mol.GetAtoms():
                features = []
                for feature_name in self.node_features:
                    if feature_name in self.atom_features:
                        features.append(self.atom_features[feature_name](atom))
                    else:
                        logger.warning(f"Unknown node feature: {feature_name}")
                        features.append(0)
                node_features.append(features)
            
            # Extract edge features and connectivity
            edge_indices = []
            edge_features = []
            
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                # Add both directions for undirected graph
                edge_indices.extend([[i, j], [j, i]])
                
                # Extract bond features
                features = []
                for feature_name in self.edge_features:
                    if feature_name in self.bond_features:
                        features.append(self.bond_features[feature_name](bond))
                    else:
                        logger.warning(f"Unknown edge feature: {feature_name}")
                        features.append(0)
                
                # Add features for both directions
                edge_features.extend([features, features])
            
            # Convert to tensors
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float) if edge_features else None
            
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            
        except Exception as e:
            logger.warning(f"Failed to convert SMILES to graph: {smiles}, error: {e}")
            return None
    
    def batch_smiles_to_graphs(self, smiles_list: List[str]) -> List[Optional[Data]]:
        """
        Convert batch of SMILES to graphs.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            List of PyTorch Geometric Data objects
        """
        graphs = []
        for smiles in smiles_list:
            graph = self.smiles_to_graph(smiles)
            graphs.append(graph)
        
        return graphs


class SubstructureCounterfactualAnalyzer:
    """Analyze causal substructures using counterfactual reasoning."""
    
    def __init__(self, config: Config):
        """
        Initialize counterfactual analyzer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.fragment_size_range = config.get("features.chemistry.substructure_analysis.fragment_size_range", [3, 8])
        self.max_fragments = config.get("features.chemistry.substructure_analysis.max_fragments", 1000)
        self.consistency_threshold = config.get("features.chemistry.substructure_analysis.counterfactual_threshold", 0.1)
        
    def extract_fragments(self, mol: Chem.Mol) -> List[Chem.Mol]:
        """
        Extract molecular fragments of different sizes.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            List of fragment molecules
        """
        fragments = []
        
        for size in range(self.fragment_size_range[0], self.fragment_size_range[1] + 1):
            # Use BRICS decomposition for meaningful fragments
            try:
                brics_frags = AllChem.BRICSDecompose(mol, minFragmentSize=size, maxFragmentSize=size)
                for frag_smiles in brics_frags:
                    frag_mol = Chem.MolFromSmiles(frag_smiles)
                    if frag_mol and frag_mol.GetNumHeavyAtoms() >= size:
                        fragments.append(frag_mol)
                        
                        if len(fragments) >= self.max_fragments:
                            return fragments
            except:
                continue
        
        return fragments
    
    def compute_counterfactual_scores(
        self, 
        smiles_list: List[str], 
        labels: np.ndarray,
        moa_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute counterfactual importance scores for substructures.
        
        Args:
            smiles_list: List of SMILES strings
            labels: Binary label matrix (n_compounds x n_moas)
            moa_names: List of MoA names
            
        Returns:
            Dictionary mapping MoA to fragment importance scores
        """
        logger.info("Computing counterfactual substructure scores...")
        
        # Extract all unique fragments
        all_fragments = set()
        mol_fragments = {}
        
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
                
            fragments = self.extract_fragments(mol)
            mol_fragments[i] = [Chem.MolToSmiles(frag) for frag in fragments]
            all_fragments.update(mol_fragments[i])
        
        all_fragments = list(all_fragments)
        logger.info(f"Extracted {len(all_fragments)} unique fragments")
        
        # Create fragment presence matrix
        fragment_matrix = np.zeros((len(smiles_list), len(all_fragments)))
        for i, fragments in mol_fragments.items():
            for frag in fragments:
                if frag in all_fragments:
                    j = all_fragments.index(frag)
                    fragment_matrix[i, j] = 1
        
        # Compute counterfactual scores for each MoA
        counterfactual_scores = {}
        
        for moa_idx, moa_name in enumerate(moa_names):
            moa_labels = labels[:, moa_idx]
            
            if moa_labels.sum() < 10:  # Skip MoAs with too few positive examples
                continue
            
            # Train baseline model
            baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
            baseline_scores = cross_val_score(baseline_model, fragment_matrix, moa_labels, cv=5)
            baseline_performance = baseline_scores.mean()
            
            fragment_scores = {}
            
            # Compute importance for each fragment
            for frag_idx, fragment in enumerate(all_fragments):
                # Create counterfactual matrix (remove fragment)
                counterfactual_matrix = fragment_matrix.copy()
                counterfactual_matrix[:, frag_idx] = 0
                
                # Train counterfactual model
                cf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                cf_scores = cross_val_score(cf_model, counterfactual_matrix, moa_labels, cv=5)
                cf_performance = cf_scores.mean()
                
                # Counterfactual importance = baseline - counterfactual performance
                importance = baseline_performance - cf_performance
                fragment_scores[fragment] = importance
            
            counterfactual_scores[moa_name] = fragment_scores
        
        return counterfactual_scores
    
    def get_causal_fragments(
        self, 
        counterfactual_scores: Dict[str, Dict[str, float]], 
        top_k: int = 10
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get top causal fragments for each MoA.
        
        Args:
            counterfactual_scores: Counterfactual importance scores
            top_k: Number of top fragments to return
            
        Returns:
            Dictionary mapping MoA to top causal fragments
        """
        causal_fragments = {}
        
        for moa_name, fragment_scores in counterfactual_scores.items():
            # Sort fragments by importance score
            sorted_fragments = sorted(
                fragment_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Filter by threshold and take top k
            causal_frags = [
                (frag, score) for frag, score in sorted_fragments 
                if score > self.consistency_threshold
            ][:top_k]
            
            causal_fragments[moa_name] = causal_frags
        
        return causal_fragments


class MolecularDescriptorExtractor:
    """Extract traditional molecular descriptors."""
    
    def __init__(self, config: Config):
        """
        Initialize descriptor extractor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.descriptor_types = config.get("features.chemistry.molecular_descriptors", [])
    
    def extract_morgan_fingerprints(self, smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
        """Extract Morgan fingerprints."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(n_bits)
            
            fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            return np.array(fp)
        except:
            return np.zeros(n_bits)
    
    def extract_rdkit_descriptors(self, smiles: str) -> Dict[str, float]:
        """Extract RDKit molecular descriptors."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}
            
            descriptors = {}
            
            # Basic descriptors
            descriptors['mol_weight'] = Descriptors.MolWt(mol)
            descriptors['log_p'] = Descriptors.MolLogP(mol)
            descriptors['num_h_donors'] = Descriptors.NumHDonors(mol)
            descriptors['num_h_acceptors'] = Descriptors.NumHAcceptors(mol)
            descriptors['tpsa'] = Descriptors.TPSA(mol)
            descriptors['num_rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
            descriptors['num_aromatic_rings'] = Descriptors.NumAromaticRings(mol)
            descriptors['num_heavy_atoms'] = mol.GetNumHeavyAtoms()
            descriptors['fraction_csp3'] = Descriptors.FractionCsp3(mol)
            descriptors['num_rings'] = Descriptors.RingCount(mol)
            
            return descriptors
        except:
            return {}
    
    def extract_maccs_keys(self, smiles: str) -> np.ndarray:
        """Extract MACCS keys."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(167)
            
            maccs = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
            return np.array(maccs)
        except:
            return np.zeros(167)
    
    def extract_all_descriptors(self, smiles: str) -> Dict[str, Union[np.ndarray, float]]:
        """Extract all configured descriptors."""
        descriptors = {}
        
        if "morgan_fingerprints" in self.descriptor_types:
            descriptors["morgan_fp"] = self.extract_morgan_fingerprints(smiles)
        
        if "rdkit_descriptors" in self.descriptor_types:
            rdkit_desc = self.extract_rdkit_descriptors(smiles)
            descriptors.update(rdkit_desc)
        
        if "maccs_keys" in self.descriptor_types:
            descriptors["maccs"] = self.extract_maccs_keys(smiles)
        
        return descriptors


class ChemicalFeatureExtractor:
    """Main chemical feature extractor combining all approaches."""
    
    def __init__(self, config: Config):
        """
        Initialize chemical feature extractor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.graph_featurizer = MolecularGraphFeaturizer(config)
        self.counterfactual_analyzer = SubstructureCounterfactualAnalyzer(config)
        self.descriptor_extractor = MolecularDescriptorExtractor(config)
        
        self.enable_counterfactual = config.get("features.chemistry.substructure_analysis.enable_counterfactual", True)
    
    def extract_features(
        self, 
        smiles_list: List[str], 
        labels: Optional[np.ndarray] = None,
        moa_names: Optional[List[str]] = None
    ) -> Dict[str, Union[List, Dict]]:
        """
        Extract all chemical features.
        
        Args:
            smiles_list: List of SMILES strings
            labels: Optional binary label matrix for counterfactual analysis
            moa_names: Optional list of MoA names
            
        Returns:
            Dictionary containing all extracted features
        """
        logger.info(f"Extracting chemical features for {len(smiles_list)} compounds...")
        
        features = {}
        
        # Extract molecular graphs
        logger.info("Extracting molecular graphs...")
        graphs = self.graph_featurizer.batch_smiles_to_graphs(smiles_list)
        features["molecular_graphs"] = graphs
        
        # Extract molecular descriptors
        logger.info("Extracting molecular descriptors...")
        descriptors = []
        for smiles in smiles_list:
            desc = self.descriptor_extractor.extract_all_descriptors(smiles)
            descriptors.append(desc)
        features["molecular_descriptors"] = descriptors
        
        # Extract counterfactual substructure scores
        if self.enable_counterfactual and labels is not None and moa_names is not None:
            logger.info("Computing counterfactual substructure scores...")
            counterfactual_scores = self.counterfactual_analyzer.compute_counterfactual_scores(
                smiles_list, labels, moa_names
            )
            causal_fragments = self.counterfactual_analyzer.get_causal_fragments(counterfactual_scores)
            features["counterfactual_scores"] = counterfactual_scores
            features["causal_fragments"] = causal_fragments
        
        logger.info("Chemical feature extraction completed")
        return features
