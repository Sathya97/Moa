"""
Knowledge discovery applications for MoA prediction models.

This module implements tools for discovering new drug-pathway relationships,
generating biological hypotheses, and extracting novel insights from MoA predictions.
"""

from typing import Dict, List, Tuple, Optional, Union, Any
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import fisher_exact, chi2_contingency
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import defaultdict, Counter

from moa.utils.config import Config
from moa.utils.logger import get_logger

logger = get_logger(__name__)


class KnowledgeDiscovery:
    """
    Knowledge discovery system for MoA prediction models.
    
    Discovers novel insights including:
    - New drug-pathway associations
    - MoA clustering and relationships
    - Biological hypothesis generation
    - Pathway crosstalk analysis
    - Drug polypharmacology patterns
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        moa_classes: List[str],
        pathway_annotations: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initialize knowledge discovery system.
        
        Args:
            model: Trained MoA prediction model
            config: Configuration object
            moa_classes: List of MoA class names
            pathway_annotations: Optional pathway annotations for MoAs
        """
        self.model = model
        self.config = config
        self.moa_classes = moa_classes
        self.pathway_annotations = pathway_annotations or {}
        
        # Discovery parameters
        self.significance_threshold = config.get('applications.discovery.significance_threshold', 0.05)
        self.min_support = config.get('applications.discovery.min_support', 5)
        self.clustering_eps = config.get('applications.discovery.clustering_eps', 0.3)
        self.min_cluster_size = config.get('applications.discovery.min_cluster_size', 3)
        
        # Knowledge base
        self.discovered_associations = []
        self.moa_clusters = {}
        self.pathway_networks = {}
        
        self.model.eval()
        logger.info("Knowledge discovery system initialized")
    
    def discover_novel_associations(
        self,
        compound_data_list: List[Dict[str, torch.Tensor]],
        compound_metadata: List[Dict[str, Any]],
        known_associations: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Discover novel drug-pathway associations.
        
        Args:
            compound_data_list: List of compound data
            compound_metadata: Metadata for each compound
            known_associations: Known drug-pathway associations
            
        Returns:
            Dictionary containing discovered associations
        """
        logger.info(f"Discovering novel associations for {len(compound_data_list)} compounds")
        
        # Get MoA predictions for all compounds
        moa_predictions = self._get_batch_moa_predictions(compound_data_list)
        
        # Create compound-MoA matrix
        compound_moa_matrix = np.array(moa_predictions)
        
        # Discover associations using multiple methods
        association_results = {
            'statistical_associations': self._discover_statistical_associations(
                compound_moa_matrix, compound_metadata
            ),
            'clustering_associations': self._discover_clustering_associations(
                compound_moa_matrix, compound_metadata
            ),
            'pathway_enrichment': self._discover_pathway_enrichment(
                compound_moa_matrix, compound_metadata
            ),
            'novel_predictions': self._identify_novel_predictions(
                compound_moa_matrix, compound_metadata, known_associations
            )
        }
        
        # Validate and rank discoveries
        validated_associations = self._validate_associations(association_results)
        
        # Generate biological hypotheses
        hypotheses = self._generate_biological_hypotheses(validated_associations)
        
        discovery_results = {
            'total_compounds': len(compound_data_list),
            'association_methods': association_results,
            'validated_associations': validated_associations,
            'biological_hypotheses': hypotheses,
            'discovery_statistics': self._compute_discovery_statistics(association_results)
        }
        
        return discovery_results
    
    def _get_batch_moa_predictions(
        self,
        compound_data_list: List[Dict[str, torch.Tensor]]
    ) -> List[np.ndarray]:
        """Get MoA predictions for a batch of compounds."""
        predictions = []
        
        with torch.no_grad():
            for compound_data in compound_data_list:
                output = self.model(compound_data)
                pred = torch.sigmoid(output['logits']).cpu().numpy().squeeze()
                predictions.append(pred)
        
        return predictions
    
    def _discover_statistical_associations(
        self,
        compound_moa_matrix: np.ndarray,
        compound_metadata: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Discover associations using statistical tests."""
        associations = []
        
        # Extract compound properties for analysis
        compound_properties = self._extract_compound_properties(compound_metadata)
        
        for prop_name, prop_values in compound_properties.items():
            if len(set(prop_values)) < 2:  # Skip if no variation
                continue
            
            # Test association with each MoA
            for moa_idx, moa_name in enumerate(self.moa_classes):
                moa_scores = compound_moa_matrix[:, moa_idx]
                
                # Perform statistical test based on property type
                if self._is_categorical(prop_values):
                    p_value, effect_size = self._test_categorical_association(
                        prop_values, moa_scores
                    )
                else:
                    p_value, effect_size = self._test_continuous_association(
                        prop_values, moa_scores
                    )
                
                if p_value < self.significance_threshold:
                    associations.append({
                        'property': prop_name,
                        'moa': moa_name,
                        'moa_index': moa_idx,
                        'p_value': p_value,
                        'effect_size': effect_size,
                        'association_type': 'statistical',
                        'significance': 'high' if p_value < 0.01 else 'medium'
                    })
        
        return sorted(associations, key=lambda x: x['p_value'])
    
    def _discover_clustering_associations(
        self,
        compound_moa_matrix: np.ndarray,
        compound_metadata: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Discover associations through clustering analysis."""
        associations = []
        
        # Cluster compounds based on MoA profiles
        clusters = self._cluster_compounds(compound_moa_matrix)
        
        # Analyze each cluster for enriched properties
        for cluster_id in set(clusters):
            if cluster_id == -1:  # Skip noise points
                continue
            
            cluster_indices = np.where(clusters == cluster_id)[0]
            if len(cluster_indices) < self.min_cluster_size:
                continue
            
            # Get cluster MoA profile
            cluster_moa_profile = np.mean(compound_moa_matrix[cluster_indices], axis=0)
            top_moas = np.argsort(cluster_moa_profile)[-5:][::-1]
            
            # Test for enriched compound properties in this cluster
            cluster_metadata = [compound_metadata[i] for i in cluster_indices]
            enriched_properties = self._find_enriched_properties(
                cluster_metadata, compound_metadata
            )
            
            for prop_name, enrichment_data in enriched_properties.items():
                associations.append({
                    'cluster_id': cluster_id,
                    'cluster_size': len(cluster_indices),
                    'top_moas': [self.moa_classes[i] for i in top_moas],
                    'enriched_property': prop_name,
                    'enrichment_score': enrichment_data['score'],
                    'p_value': enrichment_data['p_value'],
                    'association_type': 'clustering'
                })
        
        return associations
    
    def _discover_pathway_enrichment(
        self,
        compound_moa_matrix: np.ndarray,
        compound_metadata: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Discover pathway enrichment patterns."""
        enrichment_results = []
        
        if not self.pathway_annotations:
            logger.warning("No pathway annotations available for enrichment analysis")
            return enrichment_results
        
        # Group MoAs by pathways
        pathway_moa_groups = defaultdict(list)
        for moa_idx, moa_name in enumerate(self.moa_classes):
            if moa_name in self.pathway_annotations:
                for pathway in self.pathway_annotations[moa_name]:
                    pathway_moa_groups[pathway].append(moa_idx)
        
        # Test pathway enrichment for different compound groups
        compound_groups = self._group_compounds_by_properties(compound_metadata)
        
        for group_name, group_indices in compound_groups.items():
            if len(group_indices) < self.min_support:
                continue
            
            group_moa_scores = compound_moa_matrix[group_indices]
            background_moa_scores = compound_moa_matrix
            
            for pathway_name, moa_indices in pathway_moa_groups.items():
                if len(moa_indices) < 2:
                    continue
                
                # Compute pathway activity scores
                group_pathway_scores = np.mean(group_moa_scores[:, moa_indices], axis=1)
                background_pathway_scores = np.mean(background_moa_scores[:, moa_indices], axis=1)
                
                # Test for enrichment
                p_value, effect_size = self._test_enrichment(
                    group_pathway_scores, background_pathway_scores
                )
                
                if p_value < self.significance_threshold:
                    enrichment_results.append({
                        'compound_group': group_name,
                        'pathway': pathway_name,
                        'moa_count': len(moa_indices),
                        'group_size': len(group_indices),
                        'p_value': p_value,
                        'effect_size': effect_size,
                        'mean_pathway_score': np.mean(group_pathway_scores),
                        'association_type': 'pathway_enrichment'
                    })
        
        return sorted(enrichment_results, key=lambda x: x['p_value'])
    
    def _identify_novel_predictions(
        self,
        compound_moa_matrix: np.ndarray,
        compound_metadata: List[Dict[str, Any]],
        known_associations: Optional[Dict[str, List[str]]] = None
    ) -> List[Dict[str, Any]]:
        """Identify novel predictions not in known associations."""
        novel_predictions = []
        
        if not known_associations:
            logger.warning("No known associations provided - all predictions considered novel")
            known_associations = {}
        
        for i, compound_meta in enumerate(compound_metadata):
            compound_id = compound_meta.get('compound_id', f'compound_{i}')
            compound_name = compound_meta.get('compound_name', compound_id)
            
            # Get predicted MoAs for this compound
            moa_scores = compound_moa_matrix[i]
            predicted_moas = []
            
            for moa_idx, score in enumerate(moa_scores):
                if score > 0.5:  # Threshold for positive prediction
                    moa_name = self.moa_classes[moa_idx]
                    predicted_moas.append((moa_name, score))
            
            # Check which predictions are novel
            known_moas = set(known_associations.get(compound_id, []))
            
            for moa_name, score in predicted_moas:
                if moa_name not in known_moas:
                    novel_predictions.append({
                        'compound_id': compound_id,
                        'compound_name': compound_name,
                        'predicted_moa': moa_name,
                        'prediction_score': score,
                        'confidence': 'high' if score > 0.8 else 'medium' if score > 0.6 else 'low',
                        'association_type': 'novel_prediction'
                    })
        
        return sorted(novel_predictions, key=lambda x: x['prediction_score'], reverse=True)
    
    def _extract_compound_properties(
        self,
        compound_metadata: List[Dict[str, Any]]
    ) -> Dict[str, List[Any]]:
        """Extract compound properties for analysis."""
        properties = defaultdict(list)
        
        for metadata in compound_metadata:
            for key, value in metadata.items():
                if key not in ['compound_id', 'compound_name']:  # Skip identifiers
                    properties[key].append(value)
        
        # Filter properties with sufficient data
        filtered_properties = {}
        for prop_name, values in properties.items():
            non_null_values = [v for v in values if v is not None and v != '']
            if len(non_null_values) >= self.min_support:
                filtered_properties[prop_name] = values
        
        return filtered_properties
    
    def _is_categorical(self, values: List[Any]) -> bool:
        """Check if values represent a categorical variable."""
        unique_values = set(v for v in values if v is not None)
        return len(unique_values) <= 20 or any(isinstance(v, str) for v in unique_values)
    
    def _test_categorical_association(
        self,
        categories: List[Any],
        moa_scores: np.ndarray
    ) -> Tuple[float, float]:
        """Test association between categorical variable and MoA scores."""
        # Convert to binary high/low MoA activity
        high_activity = moa_scores > np.median(moa_scores)
        
        # Create contingency table
        unique_categories = list(set(c for c in categories if c is not None))
        if len(unique_categories) < 2:
            return 1.0, 0.0
        
        contingency_table = []
        for category in unique_categories:
            category_indices = [i for i, c in enumerate(categories) if c == category]
            high_count = sum(high_activity[category_indices])
            low_count = len(category_indices) - high_count
            contingency_table.append([high_count, low_count])
        
        # Perform chi-square test
        try:
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            # Compute Cramér's V as effect size
            n = sum(sum(row) for row in contingency_table)
            effect_size = np.sqrt(chi2 / (n * (min(len(contingency_table), 2) - 1)))
        except:
            p_value, effect_size = 1.0, 0.0
        
        return p_value, effect_size
    
    def _test_continuous_association(
        self,
        values: List[float],
        moa_scores: np.ndarray
    ) -> Tuple[float, float]:
        """Test association between continuous variable and MoA scores."""
        # Filter out None values
        valid_indices = [i for i, v in enumerate(values) if v is not None]
        if len(valid_indices) < self.min_support:
            return 1.0, 0.0
        
        valid_values = np.array([values[i] for i in valid_indices])
        valid_moa_scores = moa_scores[valid_indices]
        
        # Compute correlation
        try:
            correlation = np.corrcoef(valid_values, valid_moa_scores)[0, 1]
            
            # Compute p-value using t-test
            n = len(valid_values)
            t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
            from scipy.stats import t
            p_value = 2 * (1 - t.cdf(abs(t_stat), n - 2))
            
            effect_size = abs(correlation)
        except:
            p_value, effect_size = 1.0, 0.0
        
        return p_value, effect_size
    
    def _cluster_compounds(self, compound_moa_matrix: np.ndarray) -> np.ndarray:
        """Cluster compounds based on MoA profiles."""
        # Use DBSCAN for clustering
        clustering = DBSCAN(eps=self.clustering_eps, min_samples=self.min_cluster_size)
        clusters = clustering.fit_predict(compound_moa_matrix)
        
        return clusters
    
    def _find_enriched_properties(
        self,
        cluster_metadata: List[Dict[str, Any]],
        all_metadata: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """Find properties enriched in a cluster."""
        enriched_properties = {}
        
        # Extract properties for cluster and background
        cluster_properties = self._extract_compound_properties(cluster_metadata)
        background_properties = self._extract_compound_properties(all_metadata)
        
        for prop_name in cluster_properties:
            if prop_name not in background_properties:
                continue
            
            cluster_values = cluster_properties[prop_name]
            background_values = background_properties[prop_name]
            
            if self._is_categorical(cluster_values):
                # Test categorical enrichment
                enrichment_score, p_value = self._test_categorical_enrichment(
                    cluster_values, background_values
                )
            else:
                # Test continuous enrichment
                enrichment_score, p_value = self._test_continuous_enrichment(
                    cluster_values, background_values
                )
            
            if p_value < self.significance_threshold:
                enriched_properties[prop_name] = {
                    'score': enrichment_score,
                    'p_value': p_value
                }
        
        return enriched_properties
    
    def _test_categorical_enrichment(
        self,
        cluster_values: List[Any],
        background_values: List[Any]
    ) -> Tuple[float, float]:
        """Test enrichment of categorical values in cluster."""
        cluster_counts = Counter(v for v in cluster_values if v is not None)
        background_counts = Counter(v for v in background_values if v is not None)
        
        if not cluster_counts or not background_counts:
            return 0.0, 1.0
        
        # Find most enriched category
        max_enrichment = 0.0
        min_p_value = 1.0
        
        for category in cluster_counts:
            if category not in background_counts:
                continue
            
            # Fisher's exact test
            cluster_pos = cluster_counts[category]
            cluster_neg = sum(cluster_counts.values()) - cluster_pos
            background_pos = background_counts[category]
            background_neg = sum(background_counts.values()) - background_pos
            
            try:
                odds_ratio, p_value = fisher_exact([
                    [cluster_pos, cluster_neg],
                    [background_pos, background_neg]
                ])
                
                enrichment = np.log2(odds_ratio) if odds_ratio > 0 else 0
                
                if enrichment > max_enrichment:
                    max_enrichment = enrichment
                    min_p_value = p_value
            except:
                continue
        
        return max_enrichment, min_p_value
    
    def _test_continuous_enrichment(
        self,
        cluster_values: List[float],
        background_values: List[float]
    ) -> Tuple[float, float]:
        """Test enrichment of continuous values in cluster."""
        cluster_vals = np.array([v for v in cluster_values if v is not None])
        background_vals = np.array([v for v in background_values if v is not None])
        
        if len(cluster_vals) < 2 or len(background_vals) < 2:
            return 0.0, 1.0
        
        # Mann-Whitney U test
        from scipy.stats import mannwhitneyu
        try:
            statistic, p_value = mannwhitneyu(cluster_vals, background_vals, alternative='two-sided')
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(cluster_vals) - 1) * np.var(cluster_vals) + 
                                 (len(background_vals) - 1) * np.var(background_vals)) / 
                                (len(cluster_vals) + len(background_vals) - 2))
            
            effect_size = abs(np.mean(cluster_vals) - np.mean(background_vals)) / pooled_std
        except:
            effect_size, p_value = 0.0, 1.0
        
        return effect_size, p_value
    
    def _group_compounds_by_properties(
        self,
        compound_metadata: List[Dict[str, Any]]
    ) -> Dict[str, List[int]]:
        """Group compounds by their properties."""
        groups = defaultdict(list)
        
        # Group by categorical properties
        for prop_name in ['drug_class', 'indication', 'target_class']:
            for i, metadata in enumerate(compound_metadata):
                if prop_name in metadata and metadata[prop_name]:
                    group_name = f"{prop_name}_{metadata[prop_name]}"
                    groups[group_name].append(i)
        
        # Filter groups by minimum size
        filtered_groups = {name: indices for name, indices in groups.items() 
                          if len(indices) >= self.min_support}
        
        return filtered_groups
    
    def _test_enrichment(
        self,
        group_scores: np.ndarray,
        background_scores: np.ndarray
    ) -> Tuple[float, float]:
        """Test for enrichment of scores in group vs background."""
        from scipy.stats import mannwhitneyu
        
        try:
            statistic, p_value = mannwhitneyu(group_scores, background_scores, alternative='greater')
            
            # Effect size
            effect_size = (np.mean(group_scores) - np.mean(background_scores)) / np.std(background_scores)
        except:
            p_value, effect_size = 1.0, 0.0
        
        return p_value, effect_size
    
    def _validate_associations(
        self,
        association_results: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Validate and rank discovered associations."""
        all_associations = []
        
        # Collect all associations
        for method_name, associations in association_results.items():
            for assoc in associations:
                assoc['discovery_method'] = method_name
                all_associations.append(assoc)
        
        # Rank by significance and effect size
        validated_associations = []
        for assoc in all_associations:
            # Compute validation score
            p_value = assoc.get('p_value', 1.0)
            effect_size = assoc.get('effect_size', 0.0)
            
            validation_score = -np.log10(p_value) * effect_size
            assoc['validation_score'] = validation_score
            
            # Add confidence level
            if p_value < 0.001 and effect_size > 0.5:
                assoc['confidence'] = 'high'
            elif p_value < 0.01 and effect_size > 0.3:
                assoc['confidence'] = 'medium'
            else:
                assoc['confidence'] = 'low'
            
            validated_associations.append(assoc)
        
        # Sort by validation score
        validated_associations.sort(key=lambda x: x['validation_score'], reverse=True)
        
        return validated_associations
    
    def _generate_biological_hypotheses(
        self,
        validated_associations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate biological hypotheses from discovered associations."""
        hypotheses = []
        
        # Group associations by type
        association_groups = defaultdict(list)
        for assoc in validated_associations[:50]:  # Top 50 associations
            assoc_type = assoc.get('association_type', 'unknown')
            association_groups[assoc_type].append(assoc)
        
        # Generate hypotheses for each type
        for assoc_type, associations in association_groups.items():
            if assoc_type == 'statistical':
                hypotheses.extend(self._generate_statistical_hypotheses(associations))
            elif assoc_type == 'clustering':
                hypotheses.extend(self._generate_clustering_hypotheses(associations))
            elif assoc_type == 'pathway_enrichment':
                hypotheses.extend(self._generate_pathway_hypotheses(associations))
            elif assoc_type == 'novel_prediction':
                hypotheses.extend(self._generate_novel_hypotheses(associations))
        
        return hypotheses
    
    def _generate_statistical_hypotheses(
        self,
        associations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate hypotheses from statistical associations."""
        hypotheses = []
        
        for assoc in associations[:10]:  # Top 10
            hypothesis_text = (
                f"Compounds with {assoc['property']} may preferentially target "
                f"{assoc['moa']} pathway (p={assoc['p_value']:.3e}, "
                f"effect size={assoc['effect_size']:.3f})"
            )
            
            hypotheses.append({
                'hypothesis_text': hypothesis_text,
                'hypothesis_type': 'property_moa_association',
                'confidence': assoc.get('confidence', 'medium'),
                'supporting_evidence': assoc,
                'testable_prediction': f"New compounds with {assoc['property']} should show activity against {assoc['moa']}"
            })
        
        return hypotheses
    
    def _generate_clustering_hypotheses(
        self,
        associations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate hypotheses from clustering associations."""
        hypotheses = []
        
        for assoc in associations[:5]:  # Top 5
            top_moas_text = ', '.join(assoc['top_moas'][:3])
            
            hypothesis_text = (
                f"Compounds enriched in {assoc['enriched_property']} form a distinct "
                f"pharmacological class targeting {top_moas_text} pathways"
            )
            
            hypotheses.append({
                'hypothesis_text': hypothesis_text,
                'hypothesis_type': 'pharmacological_class',
                'confidence': 'medium',
                'supporting_evidence': assoc,
                'testable_prediction': f"New compounds with {assoc['enriched_property']} should cluster with this group"
            })
        
        return hypotheses
    
    def _generate_pathway_hypotheses(
        self,
        associations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate hypotheses from pathway enrichment."""
        hypotheses = []
        
        for assoc in associations[:5]:  # Top 5
            hypothesis_text = (
                f"Compounds in {assoc['compound_group']} show preferential "
                f"activity against {assoc['pathway']} pathway, suggesting "
                f"this pathway as a key therapeutic target"
            )
            
            hypotheses.append({
                'hypothesis_text': hypothesis_text,
                'hypothesis_type': 'pathway_targeting',
                'confidence': 'high' if assoc['p_value'] < 0.001 else 'medium',
                'supporting_evidence': assoc,
                'testable_prediction': f"Modulating {assoc['pathway']} should be therapeutic for {assoc['compound_group']} indications"
            })
        
        return hypotheses
    
    def _generate_novel_hypotheses(
        self,
        associations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate hypotheses from novel predictions."""
        hypotheses = []
        
        for assoc in associations[:10]:  # Top 10
            hypothesis_text = (
                f"{assoc['compound_name']} may have therapeutic potential "
                f"through {assoc['predicted_moa']} mechanism "
                f"(confidence: {assoc['confidence']})"
            )
            
            hypotheses.append({
                'hypothesis_text': hypothesis_text,
                'hypothesis_type': 'novel_therapeutic_target',
                'confidence': assoc['confidence'],
                'supporting_evidence': assoc,
                'testable_prediction': f"Experimental validation of {assoc['compound_name']} activity against {assoc['predicted_moa']}"
            })
        
        return hypotheses
    
    def _compute_discovery_statistics(
        self,
        association_results: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Compute statistics for discovery results."""
        stats = {
            'total_associations': 0,
            'significant_associations': 0,
            'methods_used': list(association_results.keys()),
            'associations_by_method': {},
            'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0}
        }
        
        for method_name, associations in association_results.items():
            stats['associations_by_method'][method_name] = len(associations)
            stats['total_associations'] += len(associations)
            
            for assoc in associations:
                if assoc.get('p_value', 1.0) < self.significance_threshold:
                    stats['significant_associations'] += 1
                
                confidence = assoc.get('confidence', 'low')
                if confidence in stats['confidence_distribution']:
                    stats['confidence_distribution'][confidence] += 1
        
        return stats
