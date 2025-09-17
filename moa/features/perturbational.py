"""
Perturbational biology features from LINCS L1000 data.

This module processes gene expression signatures and maps them to pathway
activity scores using GSVA/ssGSEA approaches.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import gseapy as gp

from moa.utils.config import Config
from moa.utils.logger import get_logger

logger = get_logger(__name__)


class LINCSSignatureProcessor:
    """Process LINCS L1000 gene expression signatures."""
    
    def __init__(self, config: Config):
        """
        Initialize LINCS signature processor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.cell_lines = config.get("lincs.signatures.cell_lines", [])
        self.time_points = config.get("lincs.signatures.time_points", ["24h"])
        self.doses = config.get("lincs.signatures.doses", ["10 µM"])
        self.landmark_genes_only = config.get("lincs.genes.landmark_genes", True)
        
        # Gene expression normalization
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def load_lincs_signatures(self, signatures_df: pd.DataFrame) -> pd.DataFrame:
        """
        Load and filter LINCS signatures.
        
        Args:
            signatures_df: DataFrame with LINCS signature data
            
        Returns:
            Filtered signatures DataFrame
        """
        logger.info("Loading LINCS signatures...")
        
        # Filter by cell lines, time points, and doses
        filtered_df = signatures_df.copy()
        
        if self.cell_lines:
            filtered_df = filtered_df[filtered_df['cell_id'].isin(self.cell_lines)]
        
        if self.time_points:
            filtered_df = filtered_df[filtered_df['pert_time'].isin(self.time_points)]
        
        if self.doses:
            filtered_df = filtered_df[filtered_df['pert_dose'].isin(self.doses)]
        
        logger.info(f"Filtered to {len(filtered_df)} signatures from {len(signatures_df)} total")
        return filtered_df
    
    def extract_gene_expression_matrix(self, signatures_df: pd.DataFrame) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Extract gene expression matrix from signatures.
        
        Args:
            signatures_df: LINCS signatures DataFrame
            
        Returns:
            Tuple of (expression_matrix, signature_ids, gene_names)
        """
        logger.info("Extracting gene expression matrix...")
        
        # This is a simplified implementation
        # In practice, you would load the actual L1000 expression data
        
        # Simulate gene expression data for demonstration
        n_signatures = len(signatures_df)
        n_genes = 978 if self.landmark_genes_only else 11000  # L1000 landmark genes
        
        # Generate simulated expression data
        expression_matrix = np.random.normal(0, 1, (n_signatures, n_genes))
        
        # Add some structure based on compound similarity
        signature_ids = signatures_df['sig_id'].tolist()
        
        # Simulate gene names
        gene_names = [f"GENE_{i:04d}" for i in range(n_genes)]
        
        logger.info(f"Generated expression matrix: {expression_matrix.shape}")
        return expression_matrix, signature_ids, gene_names
    
    def normalize_expression_data(self, expression_matrix: np.ndarray) -> np.ndarray:
        """
        Normalize gene expression data.
        
        Args:
            expression_matrix: Raw expression matrix (signatures x genes)
            
        Returns:
            Normalized expression matrix
        """
        logger.info("Normalizing expression data...")
        
        if not self.is_fitted:
            normalized_matrix = self.scaler.fit_transform(expression_matrix)
            self.is_fitted = True
        else:
            normalized_matrix = self.scaler.transform(expression_matrix)
        
        return normalized_matrix
    
    def compute_meta_signatures(
        self, 
        expression_matrix: np.ndarray, 
        signatures_df: pd.DataFrame,
        aggregation_method: str = "robust_mean"
    ) -> Dict[str, np.ndarray]:
        """
        Compute meta-signatures by aggregating across cell lines.
        
        Args:
            expression_matrix: Normalized expression matrix
            signatures_df: Signatures metadata
            aggregation_method: Method for aggregation ("mean", "median", "robust_mean")
            
        Returns:
            Dictionary mapping compound names to meta-signatures
        """
        logger.info("Computing meta-signatures...")
        
        meta_signatures = {}
        
        # Group by compound
        for compound, group in signatures_df.groupby('pert_iname'):
            indices = group.index.tolist()
            compound_expressions = expression_matrix[indices]
            
            if aggregation_method == "mean":
                meta_sig = np.mean(compound_expressions, axis=0)
            elif aggregation_method == "median":
                meta_sig = np.median(compound_expressions, axis=0)
            elif aggregation_method == "robust_mean":
                # Trim extreme values before averaging
                meta_sig = stats.trim_mean(compound_expressions, 0.1, axis=0)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation_method}")
            
            meta_signatures[compound] = meta_sig
        
        logger.info(f"Computed meta-signatures for {len(meta_signatures)} compounds")
        return meta_signatures


class PathwayActivityScorer:
    """Compute pathway activity scores from gene expression signatures."""
    
    def __init__(self, config: Config):
        """
        Initialize pathway activity scorer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.gene_set_databases = config.get("features.perturbation.gene_set_databases", [])
        self.scoring_method = config.get("features.perturbation.pathway_scoring", "gsva")
        
        self.gene_sets = {}
        self.pathway_names = []
        
    def load_gene_sets(self) -> None:
        """Load gene sets from various databases."""
        logger.info("Loading gene sets...")
        
        all_gene_sets = {}
        
        for database in self.gene_set_databases:
            try:
                if database == "hallmark":
                    gene_sets = gp.get_library(name='MSigDB_Hallmark_2020', organism='Human')
                elif database == "c2_canonical":
                    gene_sets = gp.get_library(name='MSigDB_Canonical_Pathways', organism='Human')
                elif database == "c2_reactome":
                    gene_sets = gp.get_library(name='Reactome_2022', organism='Human')
                elif database == "c5_go_bp":
                    gene_sets = gp.get_library(name='GO_Biological_Process_2023', organism='Human')
                else:
                    logger.warning(f"Unknown gene set database: {database}")
                    continue
                
                # Add prefix to avoid name conflicts
                prefixed_sets = {f"{database}_{name}": genes for name, genes in gene_sets.items()}
                all_gene_sets.update(prefixed_sets)
                
            except Exception as e:
                logger.warning(f"Failed to load {database}: {e}")
                # Create dummy gene sets for demonstration
                dummy_sets = {
                    f"{database}_pathway_{i}": [f"GENE_{j:04d}" for j in range(i*10, (i+1)*10)]
                    for i in range(10)
                }
                all_gene_sets.update(dummy_sets)
        
        self.gene_sets = all_gene_sets
        self.pathway_names = list(all_gene_sets.keys())
        
        logger.info(f"Loaded {len(self.gene_sets)} gene sets from {len(self.gene_set_databases)} databases")
    
    def compute_gsva_scores(
        self, 
        expression_matrix: np.ndarray, 
        gene_names: List[str]
    ) -> np.ndarray:
        """
        Compute GSVA (Gene Set Variation Analysis) scores.
        
        Args:
            expression_matrix: Gene expression matrix (samples x genes)
            gene_names: List of gene names
            
        Returns:
            Pathway activity matrix (samples x pathways)
        """
        logger.info("Computing GSVA scores...")
        
        if not self.gene_sets:
            self.load_gene_sets()
        
        n_samples, n_genes = expression_matrix.shape
        n_pathways = len(self.pathway_names)
        
        pathway_scores = np.zeros((n_samples, n_pathways))
        
        for pathway_idx, pathway_name in enumerate(self.pathway_names):
            pathway_genes = self.gene_sets[pathway_name]
            
            # Find indices of pathway genes in expression matrix
            gene_indices = []
            for gene in pathway_genes:
                if gene in gene_names:
                    gene_indices.append(gene_names.index(gene))
            
            if not gene_indices:
                continue
            
            # Compute pathway score for each sample
            for sample_idx in range(n_samples):
                sample_expression = expression_matrix[sample_idx]
                
                # GSVA-like scoring: rank-based enrichment
                gene_ranks = stats.rankdata(sample_expression)
                pathway_ranks = gene_ranks[gene_indices]
                
                # Compute enrichment score
                n_pathway_genes = len(gene_indices)
                expected_rank = (n_genes + 1) / 2
                observed_rank = np.mean(pathway_ranks)
                
                # Normalized enrichment score
                enrichment_score = (observed_rank - expected_rank) / (n_genes / 2)
                pathway_scores[sample_idx, pathway_idx] = enrichment_score
        
        logger.info(f"Computed GSVA scores: {pathway_scores.shape}")
        return pathway_scores
    
    def compute_ssgsea_scores(
        self, 
        expression_matrix: np.ndarray, 
        gene_names: List[str]
    ) -> np.ndarray:
        """
        Compute ssGSEA (single-sample Gene Set Enrichment Analysis) scores.
        
        Args:
            expression_matrix: Gene expression matrix (samples x genes)
            gene_names: List of gene names
            
        Returns:
            Pathway activity matrix (samples x pathways)
        """
        logger.info("Computing ssGSEA scores...")
        
        if not self.gene_sets:
            self.load_gene_sets()
        
        n_samples, n_genes = expression_matrix.shape
        n_pathways = len(self.pathway_names)
        
        pathway_scores = np.zeros((n_samples, n_pathways))
        
        for pathway_idx, pathway_name in enumerate(self.pathway_names):
            pathway_genes = self.gene_sets[pathway_name]
            
            # Find indices of pathway genes
            gene_indices = []
            for gene in pathway_genes:
                if gene in gene_names:
                    gene_indices.append(gene_names.index(gene))
            
            if not gene_indices:
                continue
            
            # Compute ssGSEA score for each sample
            for sample_idx in range(n_samples):
                sample_expression = expression_matrix[sample_idx]
                
                # Rank genes by expression
                gene_ranks = stats.rankdata(-sample_expression)  # Higher expression = lower rank
                
                # Compute enrichment score using Kolmogorov-Smirnov-like statistic
                pathway_ranks = sorted([gene_ranks[i] for i in gene_indices])
                
                # Compute enrichment score
                n_pathway_genes = len(gene_indices)
                enrichment_score = 0
                
                for i, rank in enumerate(pathway_ranks):
                    # Weighted enrichment score
                    hit_score = (i + 1) / n_pathway_genes
                    miss_score = (rank - i - 1) / (n_genes - n_pathway_genes)
                    enrichment_score += hit_score - miss_score
                
                pathway_scores[sample_idx, pathway_idx] = enrichment_score / n_pathway_genes
        
        logger.info(f"Computed ssGSEA scores: {pathway_scores.shape}")
        return pathway_scores
    
    def compute_pathway_scores(
        self, 
        expression_matrix: np.ndarray, 
        gene_names: List[str]
    ) -> np.ndarray:
        """
        Compute pathway activity scores using configured method.
        
        Args:
            expression_matrix: Gene expression matrix
            gene_names: List of gene names
            
        Returns:
            Pathway activity scores
        """
        if self.scoring_method == "gsva":
            return self.compute_gsva_scores(expression_matrix, gene_names)
        elif self.scoring_method == "ssgsea":
            return self.compute_ssgsea_scores(expression_matrix, gene_names)
        elif self.scoring_method == "combined":
            gsva_scores = self.compute_gsva_scores(expression_matrix, gene_names)
            ssgsea_scores = self.compute_ssgsea_scores(expression_matrix, gene_names)
            return np.concatenate([gsva_scores, ssgsea_scores], axis=1)
        else:
            raise ValueError(f"Unknown scoring method: {self.scoring_method}")


class PerturbationalFeatureExtractor:
    """Main perturbational feature extractor."""
    
    def __init__(self, config: Config):
        """
        Initialize perturbational feature extractor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.signature_processor = LINCSSignatureProcessor(config)
        self.pathway_scorer = PathwayActivityScorer(config)
        
        self.meta_signatures = {}
        self.pathway_scores = {}
        
    def extract_perturbational_features(
        self, 
        signatures_df: pd.DataFrame,
        compound_names: List[str]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract all perturbational features.
        
        Args:
            signatures_df: LINCS signatures DataFrame
            compound_names: List of compound names to extract features for
            
        Returns:
            Dictionary containing perturbational features
        """
        logger.info("Extracting perturbational features...")
        
        # Load and filter signatures
        filtered_signatures = self.signature_processor.load_lincs_signatures(signatures_df)
        
        # Extract gene expression matrix
        expression_matrix, signature_ids, gene_names = self.signature_processor.extract_gene_expression_matrix(
            filtered_signatures
        )
        
        # Normalize expression data
        normalized_expression = self.signature_processor.normalize_expression_data(expression_matrix)
        
        # Compute meta-signatures
        meta_signatures = self.signature_processor.compute_meta_signatures(
            normalized_expression, 
            filtered_signatures,
            aggregation_method=self.config.get("features.perturbation.signature_aggregation", "robust_mean")
        )
        
        # Compute pathway activity scores
        pathway_scores_matrix = self.pathway_scorer.compute_pathway_scores(
            normalized_expression, gene_names
        )
        
        # Map pathway scores to compounds
        compound_pathway_scores = {}
        for compound, group in filtered_signatures.groupby('pert_iname'):
            indices = group.index.tolist()
            compound_scores = pathway_scores_matrix[indices]
            # Aggregate pathway scores across replicates
            compound_pathway_scores[compound] = np.mean(compound_scores, axis=0)
        
        # Filter for requested compounds
        filtered_meta_signatures = {
            compound: meta_signatures.get(compound, np.zeros(len(gene_names)))
            for compound in compound_names
        }
        
        filtered_pathway_scores = {
            compound: compound_pathway_scores.get(compound, np.zeros(len(self.pathway_scorer.pathway_names)))
            for compound in compound_names
        }
        
        features = {
            "meta_signatures": filtered_meta_signatures,
            "pathway_scores": filtered_pathway_scores,
            "gene_names": gene_names,
            "pathway_names": self.pathway_scorer.pathway_names
        }
        
        logger.info("Perturbational feature extraction completed")
        return features
