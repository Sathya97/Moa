"""
Main feature extraction orchestrator for MoA prediction.

This module coordinates all feature extraction approaches and provides
a unified interface for extracting multi-modal features.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from moa.features.chemical import ChemicalFeatureExtractor
from moa.features.mechanism_tokens import MechTokenFeatureExtractor
from moa.features.perturbational import PerturbationalFeatureExtractor
from moa.features.protein_structure import ProteinPocketFeatureExtractor
from moa.utils.config import Config
from moa.utils.logger import get_logger

logger = get_logger(__name__)


class MultiModalFeatureExtractor:
    """Main feature extractor that coordinates all modalities."""
    
    def __init__(self, config: Config):
        """
        Initialize multi-modal feature extractor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.enabled_modalities = config.get("scope.modalities", {})
        
        # Initialize feature extractors based on enabled modalities
        self.extractors = {}
        
        if self.enabled_modalities.get("chemistry", False):
            self.extractors["chemistry"] = ChemicalFeatureExtractor(config)
            logger.info("Initialized chemical feature extractor")
        
        if self.enabled_modalities.get("targets", False) or self.enabled_modalities.get("pathways", False):
            self.extractors["mechanism_tokens"] = MechTokenFeatureExtractor(config)
            logger.info("Initialized mechanism token extractor")
        
        if self.enabled_modalities.get("perturbation", False):
            self.extractors["perturbation"] = PerturbationalFeatureExtractor(config)
            logger.info("Initialized perturbational feature extractor")
        
        if self.enabled_modalities.get("structures", False):
            self.extractors["protein_structure"] = ProteinPocketFeatureExtractor(config)
            logger.info("Initialized protein structure feature extractor")
    
    def extract_all_features(
        self,
        compound_data: pd.DataFrame,
        data_sources: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict[str, Dict]:
        """
        Extract features from all enabled modalities.
        
        Args:
            compound_data: DataFrame with compound information (SMILES, ChEMBL IDs, etc.)
            data_sources: Optional dictionary of additional data sources
            
        Returns:
            Dictionary containing features from all modalities
        """
        logger.info(f"Extracting multi-modal features for {len(compound_data)} compounds...")
        
        all_features = {}
        
        # Extract SMILES and compound IDs
        smiles_list = compound_data["standardized_smiles"].tolist()
        compound_ids = compound_data["molecule_chembl_id"].tolist()
        
        # Extract labels if available for counterfactual analysis
        labels = None
        moa_names = None
        if "moa_list" in compound_data.columns:
            # Convert MoA lists to binary matrix
            from sklearn.preprocessing import MultiLabelBinarizer
            mlb = MultiLabelBinarizer()
            labels = mlb.fit_transform(compound_data["moa_list"])
            moa_names = list(mlb.classes_)
        
        # Extract chemical features
        if "chemistry" in self.extractors:
            logger.info("Extracting chemical features...")
            chemical_features = self.extractors["chemistry"].extract_features(
                smiles_list, labels, moa_names
            )
            all_features["chemistry"] = chemical_features
        
        # Extract mechanism tokens
        if "mechanism_tokens" in self.extractors:
            logger.info("Extracting mechanism tokens...")
            
            # Build mechanism tokens if not already built
            if data_sources:
                self.extractors["mechanism_tokens"].build_mechanism_tokens(data_sources)
            
            # Extract compound-specific tokens
            compound_tokens = self.extractors["mechanism_tokens"].extract_compound_tokens(compound_ids)
            
            # Extract MoA tokens if available
            moa_tokens = {}
            if moa_names:
                moa_tokens = self.extractors["mechanism_tokens"].get_moa_tokens(moa_names)
            
            all_features["mechanism_tokens"] = {
                "compound_tokens": compound_tokens,
                "moa_tokens": moa_tokens
            }
        
        # Extract perturbational features
        if "perturbation" in self.extractors:
            logger.info("Extracting perturbational features...")
            
            if data_sources and "lincs_signatures" in data_sources:
                # Map compound ChEMBL IDs to compound names for LINCS lookup
                compound_names = []
                for chembl_id in compound_ids:
                    # This is simplified - in practice you'd have a proper mapping
                    compound_name = chembl_id.replace("CHEMBL", "compound_")
                    compound_names.append(compound_name)
                
                perturbational_features = self.extractors["perturbation"].extract_perturbational_features(
                    data_sources["lincs_signatures"], compound_names
                )
                all_features["perturbation"] = perturbational_features
            else:
                logger.warning("LINCS signatures not available for perturbational features")
        
        # Extract protein structure features
        if "protein_structure" in self.extractors:
            logger.info("Extracting protein structure features...")
            
            # Get UniProt IDs from targets (simplified mapping)
            uniprot_ids = []
            if "target_chembl_id" in compound_data.columns:
                # This is simplified - in practice you'd have proper ChEMBL->UniProt mapping
                unique_targets = compound_data["target_chembl_id"].unique()
                uniprot_ids = [f"P{target.replace('CHEMBL', '')[:5]}" for target in unique_targets if pd.notna(target)]
            
            if uniprot_ids:
                pocket_features = self.extractors["protein_structure"].extract_pocket_features(uniprot_ids)
                all_features["protein_structure"] = {"pocket_features": pocket_features}
            else:
                logger.warning("No target information available for protein structure features")
        
        logger.info("Multi-modal feature extraction completed")
        return all_features
    
    def save_features(self, features: Dict, save_dir: Path) -> None:
        """
        Save extracted features to disk.
        
        Args:
            features: Dictionary of extracted features
            save_dir: Directory to save features
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for modality, modality_features in features.items():
            modality_dir = save_dir / modality
            modality_dir.mkdir(exist_ok=True)
            
            if modality == "chemistry":
                # Save molecular graphs
                if "molecular_graphs" in modality_features:
                    torch.save(modality_features["molecular_graphs"], modality_dir / "molecular_graphs.pt")
                
                # Save molecular descriptors
                if "molecular_descriptors" in modality_features:
                    pd.DataFrame(modality_features["molecular_descriptors"]).to_pickle(
                        modality_dir / "molecular_descriptors.pkl"
                    )
                
                # Save counterfactual scores
                if "counterfactual_scores" in modality_features:
                    import json
                    with open(modality_dir / "counterfactual_scores.json", "w") as f:
                        # Convert numpy arrays to lists for JSON serialization
                        serializable_scores = {}
                        for moa, scores in modality_features["counterfactual_scores"].items():
                            serializable_scores[moa] = {frag: float(score) for frag, score in scores.items()}
                        json.dump(serializable_scores, f, indent=2)
            
            elif modality == "mechanism_tokens":
                # Save mechanism tokens
                if "compound_tokens" in modality_features:
                    np.savez(modality_dir / "compound_tokens.npz", **modality_features["compound_tokens"])
                
                if "moa_tokens" in modality_features:
                    np.savez(modality_dir / "moa_tokens.npz", **modality_features["moa_tokens"])
            
            elif modality == "perturbation":
                # Save perturbational features
                if "meta_signatures" in modality_features:
                    np.savez(modality_dir / "meta_signatures.npz", **modality_features["meta_signatures"])
                
                if "pathway_scores" in modality_features:
                    np.savez(modality_dir / "pathway_scores.npz", **modality_features["pathway_scores"])
                
                # Save gene and pathway names
                if "gene_names" in modality_features:
                    with open(modality_dir / "gene_names.txt", "w") as f:
                        f.write("\n".join(modality_features["gene_names"]))
                
                if "pathway_names" in modality_features:
                    with open(modality_dir / "pathway_names.txt", "w") as f:
                        f.write("\n".join(modality_features["pathway_names"]))
            
            elif modality == "protein_structure":
                # Save protein structure features
                if "pocket_features" in modality_features:
                    np.savez(modality_dir / "pocket_features.npz", **modality_features["pocket_features"])
        
        logger.info(f"Saved features to {save_dir}")
    
    def load_features(self, load_dir: Path) -> Dict:
        """
        Load extracted features from disk.
        
        Args:
            load_dir: Directory containing saved features
            
        Returns:
            Dictionary of loaded features
        """
        features = {}
        
        for modality_dir in load_dir.iterdir():
            if not modality_dir.is_dir():
                continue
            
            modality = modality_dir.name
            modality_features = {}
            
            if modality == "chemistry":
                # Load molecular graphs
                graphs_file = modality_dir / "molecular_graphs.pt"
                if graphs_file.exists():
                    modality_features["molecular_graphs"] = torch.load(graphs_file)
                
                # Load molecular descriptors
                descriptors_file = modality_dir / "molecular_descriptors.pkl"
                if descriptors_file.exists():
                    modality_features["molecular_descriptors"] = pd.read_pickle(descriptors_file).to_dict('records')
                
                # Load counterfactual scores
                scores_file = modality_dir / "counterfactual_scores.json"
                if scores_file.exists():
                    import json
                    with open(scores_file, "r") as f:
                        modality_features["counterfactual_scores"] = json.load(f)
            
            elif modality == "mechanism_tokens":
                # Load compound tokens
                compound_tokens_file = modality_dir / "compound_tokens.npz"
                if compound_tokens_file.exists():
                    data = np.load(compound_tokens_file)
                    modality_features["compound_tokens"] = {key: data[key] for key in data.files}
                
                # Load MoA tokens
                moa_tokens_file = modality_dir / "moa_tokens.npz"
                if moa_tokens_file.exists():
                    data = np.load(moa_tokens_file)
                    modality_features["moa_tokens"] = {key: data[key] for key in data.files}
            
            elif modality == "perturbation":
                # Load meta-signatures
                meta_sigs_file = modality_dir / "meta_signatures.npz"
                if meta_sigs_file.exists():
                    data = np.load(meta_sigs_file)
                    modality_features["meta_signatures"] = {key: data[key] for key in data.files}
                
                # Load pathway scores
                pathway_scores_file = modality_dir / "pathway_scores.npz"
                if pathway_scores_file.exists():
                    data = np.load(pathway_scores_file)
                    modality_features["pathway_scores"] = {key: data[key] for key in data.files}
                
                # Load gene and pathway names
                gene_names_file = modality_dir / "gene_names.txt"
                if gene_names_file.exists():
                    with open(gene_names_file, "r") as f:
                        modality_features["gene_names"] = f.read().strip().split("\n")
                
                pathway_names_file = modality_dir / "pathway_names.txt"
                if pathway_names_file.exists():
                    with open(pathway_names_file, "r") as f:
                        modality_features["pathway_names"] = f.read().strip().split("\n")
            
            elif modality == "protein_structure":
                # Load pocket features
                pocket_features_file = modality_dir / "pocket_features.npz"
                if pocket_features_file.exists():
                    data = np.load(pocket_features_file)
                    modality_features["pocket_features"] = {key: data[key] for key in data.files}
            
            if modality_features:
                features[modality] = modality_features
        
        logger.info(f"Loaded features from {load_dir}")
        return features


def extract_features_for_dataset(
    config: Config,
    compound_data: pd.DataFrame,
    data_sources: Optional[Dict[str, pd.DataFrame]] = None,
    save_dir: Optional[Path] = None
) -> Dict:
    """
    Convenience function to extract features for a dataset.
    
    Args:
        config: Configuration object
        compound_data: DataFrame with compound information
        data_sources: Optional dictionary of additional data sources
        save_dir: Optional directory to save features
        
    Returns:
        Dictionary of extracted features
    """
    extractor = MultiModalFeatureExtractor(config)
    features = extractor.extract_all_features(compound_data, data_sources)
    
    if save_dir:
        extractor.save_features(features, save_dir)
    
    return features
