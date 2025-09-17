"""
Drug repurposing pipeline using MoA predictions.

This module implements automated drug repurposing by identifying compounds
with similar mechanisms of action and ranking repurposing candidates.
"""

from typing import Dict, List, Tuple, Optional, Union, Any
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from moa.utils.config import Config
from moa.utils.logger import get_logger

logger = get_logger(__name__)


class DrugRepurposingPipeline:
    """
    Automated drug repurposing pipeline using MoA predictions.
    
    Identifies repurposing candidates by:
    - Computing MoA similarity between compounds
    - Ranking compounds by therapeutic potential
    - Identifying novel drug-disease associations
    - Generating repurposing hypotheses
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        moa_classes: List[str],
        compound_database: Optional[pd.DataFrame] = None
    ):
        """
        Initialize drug repurposing pipeline.
        
        Args:
            model: Trained MoA prediction model
            config: Configuration object
            moa_classes: List of MoA class names
            compound_database: Database of compounds with metadata
        """
        self.model = model
        self.config = config
        self.moa_classes = moa_classes
        self.compound_database = compound_database
        
        # Repurposing parameters
        self.similarity_threshold = config.get('applications.repurposing.similarity_threshold', 0.7)
        self.top_k_candidates = config.get('applications.repurposing.top_k_candidates', 50)
        self.min_confidence = config.get('applications.repurposing.min_confidence', 0.5)
        
        # Caches for efficiency
        self.moa_embeddings_cache = {}
        self.similarity_matrix_cache = None
        
        self.model.eval()
        logger.info("Drug repurposing pipeline initialized")
    
    def identify_repurposing_candidates(
        self,
        query_compound_data: Dict[str, torch.Tensor],
        target_disease: str,
        candidate_compounds: List[Dict[str, torch.Tensor]],
        compound_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Identify drug repurposing candidates for a target disease.
        
        Args:
            query_compound_data: Data for query compound (known effective)
            target_disease: Target disease name
            candidate_compounds: List of candidate compound data
            compound_metadata: Optional metadata for candidates
            
        Returns:
            Dictionary containing repurposing analysis results
        """
        logger.info(f"Identifying repurposing candidates for {target_disease}")
        
        # Get MoA predictions for query compound
        query_moa_profile = self._get_moa_profile(query_compound_data)
        
        # Get MoA profiles for all candidates
        candidate_profiles = []
        for candidate_data in candidate_compounds:
            profile = self._get_moa_profile(candidate_data)
            candidate_profiles.append(profile)
        
        # Compute similarities
        similarities = self._compute_moa_similarities(query_moa_profile, candidate_profiles)
        
        # Rank candidates
        ranked_candidates = self._rank_repurposing_candidates(
            similarities, candidate_profiles, compound_metadata
        )
        
        # Generate repurposing hypotheses
        hypotheses = self._generate_repurposing_hypotheses(
            query_moa_profile, ranked_candidates, target_disease
        )
        
        # Compute confidence scores
        confidence_scores = self._compute_repurposing_confidence(
            query_moa_profile, ranked_candidates
        )
        
        repurposing_results = {
            'target_disease': target_disease,
            'query_moa_profile': query_moa_profile,
            'ranked_candidates': ranked_candidates,
            'similarities': similarities,
            'hypotheses': hypotheses,
            'confidence_scores': confidence_scores,
            'summary_statistics': self._compute_summary_statistics(similarities, ranked_candidates)
        }
        
        return repurposing_results
    
    def _get_moa_profile(self, compound_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Get MoA prediction profile for a compound."""
        with torch.no_grad():
            # Get model predictions
            output = self.model(compound_data)
            predictions = torch.sigmoid(output['logits']).cpu().numpy()
            
            # Extract embeddings if available
            embeddings = None
            if 'embeddings' in output:
                embeddings = output['embeddings'].cpu().numpy()
            
            moa_profile = {
                'predictions': predictions.squeeze(),
                'embeddings': embeddings.squeeze() if embeddings is not None else None,
                'top_moas': self._get_top_moas(predictions.squeeze()),
                'moa_vector': predictions.squeeze()
            }
        
        return moa_profile
    
    def _get_top_moas(self, predictions: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Get top predicted MoAs for a compound."""
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        top_moas = []
        
        for idx in top_indices:
            top_moas.append({
                'moa_name': self.moa_classes[idx],
                'moa_index': int(idx),
                'score': float(predictions[idx]),
                'confidence': 'high' if predictions[idx] > 0.8 else 'medium' if predictions[idx] > 0.5 else 'low'
            })
        
        return top_moas
    
    def _compute_moa_similarities(
        self,
        query_profile: Dict[str, Any],
        candidate_profiles: List[Dict[str, Any]]
    ) -> List[Dict[str, float]]:
        """Compute MoA similarities between query and candidates."""
        query_vector = query_profile['moa_vector']
        similarities = []
        
        for i, candidate_profile in enumerate(candidate_profiles):
            candidate_vector = candidate_profile['moa_vector']
            
            # Compute multiple similarity metrics
            cosine_sim = 1 - cosine(query_vector, candidate_vector)
            euclidean_sim = 1 / (1 + euclidean(query_vector, candidate_vector))
            pearson_corr, _ = pearsonr(query_vector, candidate_vector)
            
            # Jaccard similarity for top MoAs
            query_top_moas = set([moa['moa_index'] for moa in query_profile['top_moas']])
            candidate_top_moas = set([moa['moa_index'] for moa in candidate_profile['top_moas']])
            jaccard_sim = len(query_top_moas & candidate_top_moas) / len(query_top_moas | candidate_top_moas)
            
            # Weighted similarity score
            weighted_similarity = (
                0.4 * cosine_sim +
                0.3 * euclidean_sim +
                0.2 * pearson_corr +
                0.1 * jaccard_sim
            )
            
            similarities.append({
                'candidate_index': i,
                'cosine_similarity': cosine_sim,
                'euclidean_similarity': euclidean_sim,
                'pearson_correlation': pearson_corr,
                'jaccard_similarity': jaccard_sim,
                'weighted_similarity': weighted_similarity
            })
        
        return similarities
    
    def _rank_repurposing_candidates(
        self,
        similarities: List[Dict[str, float]],
        candidate_profiles: List[Dict[str, Any]],
        compound_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Rank repurposing candidates by similarity and other factors."""
        ranked_candidates = []
        
        for i, sim_data in enumerate(similarities):
            candidate_idx = sim_data['candidate_index']
            candidate_profile = candidate_profiles[candidate_idx]
            
            # Base ranking score
            ranking_score = sim_data['weighted_similarity']
            
            # Adjust score based on prediction confidence
            max_prediction = np.max(candidate_profile['moa_vector'])
            confidence_bonus = 0.1 * max_prediction
            ranking_score += confidence_bonus
            
            # Adjust score based on MoA diversity
            top_moa_scores = [moa['score'] for moa in candidate_profile['top_moas']]
            diversity_penalty = 0.05 * np.std(top_moa_scores)
            ranking_score -= diversity_penalty
            
            candidate_info = {
                'candidate_index': candidate_idx,
                'ranking_score': ranking_score,
                'similarity_metrics': sim_data,
                'moa_profile': candidate_profile,
                'metadata': compound_metadata[candidate_idx] if compound_metadata else {}
            }
            
            ranked_candidates.append(candidate_info)
        
        # Sort by ranking score
        ranked_candidates.sort(key=lambda x: x['ranking_score'], reverse=True)
        
        return ranked_candidates[:self.top_k_candidates]
    
    def _generate_repurposing_hypotheses(
        self,
        query_profile: Dict[str, Any],
        ranked_candidates: List[Dict[str, Any]],
        target_disease: str
    ) -> List[Dict[str, Any]]:
        """Generate repurposing hypotheses based on MoA similarities."""
        hypotheses = []
        
        query_top_moas = [moa['moa_name'] for moa in query_profile['top_moas']]
        
        for i, candidate in enumerate(ranked_candidates[:10]):  # Top 10 candidates
            candidate_top_moas = [moa['moa_name'] for moa in candidate['moa_profile']['top_moas']]
            
            # Find shared MoAs
            shared_moas = list(set(query_top_moas) & set(candidate_top_moas))
            
            # Generate hypothesis
            hypothesis = {
                'candidate_rank': i + 1,
                'candidate_index': candidate['candidate_index'],
                'similarity_score': candidate['ranking_score'],
                'shared_moas': shared_moas,
                'hypothesis_text': self._generate_hypothesis_text(
                    shared_moas, target_disease, candidate['metadata']
                ),
                'confidence_level': self._assess_hypothesis_confidence(
                    candidate['similarity_metrics'], shared_moas
                ),
                'supporting_evidence': self._gather_supporting_evidence(
                    shared_moas, candidate['metadata']
                )
            }
            
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _generate_hypothesis_text(
        self,
        shared_moas: List[str],
        target_disease: str,
        candidate_metadata: Dict[str, Any]
    ) -> str:
        """Generate human-readable hypothesis text."""
        compound_name = candidate_metadata.get('compound_name', 'Unknown compound')
        
        if shared_moas:
            moa_text = ', '.join(shared_moas[:3])  # Top 3 shared MoAs
            hypothesis = (
                f"{compound_name} may be effective for {target_disease} "
                f"due to shared mechanisms involving {moa_text}. "
                f"The compound shows similar MoA patterns to known effective treatments."
            )
        else:
            hypothesis = (
                f"{compound_name} shows overall MoA similarity to effective treatments "
                f"for {target_disease}, suggesting potential therapeutic value through "
                f"related but distinct mechanisms."
            )
        
        return hypothesis
    
    def _assess_hypothesis_confidence(
        self,
        similarity_metrics: Dict[str, float],
        shared_moas: List[str]
    ) -> str:
        """Assess confidence level of repurposing hypothesis."""
        weighted_sim = similarity_metrics['weighted_similarity']
        num_shared_moas = len(shared_moas)
        
        if weighted_sim > 0.8 and num_shared_moas >= 3:
            return 'high'
        elif weighted_sim > 0.6 and num_shared_moas >= 2:
            return 'medium'
        elif weighted_sim > 0.4:
            return 'low'
        else:
            return 'very_low'
    
    def _gather_supporting_evidence(
        self,
        shared_moas: List[str],
        candidate_metadata: Dict[str, Any]
    ) -> List[str]:
        """Gather supporting evidence for repurposing hypothesis."""
        evidence = []
        
        # Evidence from shared MoAs
        if shared_moas:
            evidence.append(f"Shares {len(shared_moas)} key mechanisms of action")
        
        # Evidence from compound properties
        if 'drug_class' in candidate_metadata:
            evidence.append(f"Belongs to {candidate_metadata['drug_class']} drug class")
        
        if 'known_indications' in candidate_metadata:
            indications = candidate_metadata['known_indications']
            if indications:
                evidence.append(f"Has established safety profile for {len(indications)} indications")
        
        # Evidence from molecular properties
        if 'molecular_weight' in candidate_metadata:
            mw = candidate_metadata['molecular_weight']
            if 150 <= mw <= 500:
                evidence.append("Favorable molecular weight for drug-like properties")
        
        return evidence
    
    def _compute_repurposing_confidence(
        self,
        query_profile: Dict[str, Any],
        ranked_candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute overall confidence in repurposing predictions."""
        similarities = [c['ranking_score'] for c in ranked_candidates]
        
        confidence_metrics = {
            'mean_similarity': np.mean(similarities),
            'max_similarity': np.max(similarities),
            'similarity_std': np.std(similarities),
            'high_confidence_candidates': sum(1 for s in similarities if s > 0.7),
            'medium_confidence_candidates': sum(1 for s in similarities if 0.5 <= s <= 0.7),
            'low_confidence_candidates': sum(1 for s in similarities if s < 0.5)
        }
        
        # Overall confidence assessment
        if confidence_metrics['max_similarity'] > 0.8:
            overall_confidence = 'high'
        elif confidence_metrics['mean_similarity'] > 0.6:
            overall_confidence = 'medium'
        else:
            overall_confidence = 'low'
        
        confidence_metrics['overall_confidence'] = overall_confidence
        
        return confidence_metrics
    
    def _compute_summary_statistics(
        self,
        similarities: List[Dict[str, float]],
        ranked_candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute summary statistics for repurposing analysis."""
        weighted_sims = [s['weighted_similarity'] for s in similarities]
        cosine_sims = [s['cosine_similarity'] for s in similarities]
        
        summary = {
            'total_candidates': len(similarities),
            'top_candidates_analyzed': len(ranked_candidates),
            'similarity_statistics': {
                'mean_weighted_similarity': np.mean(weighted_sims),
                'std_weighted_similarity': np.std(weighted_sims),
                'max_weighted_similarity': np.max(weighted_sims),
                'min_weighted_similarity': np.min(weighted_sims),
                'mean_cosine_similarity': np.mean(cosine_sims),
                'candidates_above_threshold': sum(1 for s in weighted_sims if s > self.similarity_threshold)
            },
            'moa_diversity': self._compute_moa_diversity(ranked_candidates),
            'repurposing_potential': self._assess_repurposing_potential(weighted_sims)
        }
        
        return summary
    
    def _compute_moa_diversity(self, ranked_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute diversity of MoAs among top candidates."""
        all_moas = set()
        moa_frequencies = {}
        
        for candidate in ranked_candidates[:10]:  # Top 10
            for moa in candidate['moa_profile']['top_moas']:
                moa_name = moa['moa_name']
                all_moas.add(moa_name)
                moa_frequencies[moa_name] = moa_frequencies.get(moa_name, 0) + 1
        
        diversity_metrics = {
            'unique_moas': len(all_moas),
            'most_common_moas': sorted(moa_frequencies.items(), key=lambda x: x[1], reverse=True)[:5],
            'moa_distribution_entropy': self._compute_entropy(list(moa_frequencies.values()))
        }
        
        return diversity_metrics
    
    def _compute_entropy(self, frequencies: List[int]) -> float:
        """Compute entropy of frequency distribution."""
        total = sum(frequencies)
        if total == 0:
            return 0.0
        
        probs = [f / total for f in frequencies]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        return entropy
    
    def _assess_repurposing_potential(self, similarities: List[float]) -> str:
        """Assess overall repurposing potential."""
        high_sim_count = sum(1 for s in similarities if s > 0.7)
        medium_sim_count = sum(1 for s in similarities if 0.5 <= s <= 0.7)
        
        if high_sim_count >= 5:
            return 'excellent'
        elif high_sim_count >= 2 or medium_sim_count >= 10:
            return 'good'
        elif medium_sim_count >= 5:
            return 'moderate'
        else:
            return 'limited'
    
    def create_repurposing_network(
        self,
        repurposing_results: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> nx.Graph:
        """
        Create network visualization of repurposing relationships.
        
        Args:
            repurposing_results: Results from identify_repurposing_candidates
            output_path: Path to save network visualization
            
        Returns:
            NetworkX graph object
        """
        G = nx.Graph()
        
        # Add query compound as central node
        G.add_node('query', node_type='query', label='Query Compound')
        
        # Add candidate compounds
        for i, candidate in enumerate(repurposing_results['ranked_candidates'][:20]):
            candidate_id = f"candidate_{i}"
            similarity = candidate['ranking_score']
            
            G.add_node(
                candidate_id,
                node_type='candidate',
                similarity=similarity,
                rank=i+1,
                label=f"Candidate {i+1}"
            )
            
            # Add edge with similarity as weight
            G.add_edge('query', candidate_id, weight=similarity)
        
        # Add MoA nodes and connections
        query_moas = repurposing_results['query_moa_profile']['top_moas']
        for moa in query_moas:
            moa_id = f"moa_{moa['moa_name']}"
            G.add_node(moa_id, node_type='moa', label=moa['moa_name'], score=moa['score'])
            G.add_edge('query', moa_id, weight=moa['score'])
        
        # Visualize network if output path provided
        if output_path:
            self._visualize_repurposing_network(G, output_path)
        
        return G
    
    def _visualize_repurposing_network(self, G: nx.Graph, output_path: str) -> None:
        """Visualize repurposing network."""
        plt.figure(figsize=(15, 12))
        
        # Set up layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw different node types with different colors
        query_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'query']
        candidate_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'candidate']
        moa_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'moa']
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, nodelist=query_nodes, node_color='red', 
                              node_size=1000, alpha=0.8, label='Query')
        nx.draw_networkx_nodes(G, pos, nodelist=candidate_nodes, node_color='lightblue', 
                              node_size=500, alpha=0.7, label='Candidates')
        nx.draw_networkx_nodes(G, pos, nodelist=moa_nodes, node_color='lightgreen', 
                              node_size=300, alpha=0.6, label='MoAs')
        
        # Draw edges with weights
        edges = G.edges(data=True)
        weights = [d['weight'] for u, v, d in edges]
        nx.draw_networkx_edges(G, pos, width=[w*3 for w in weights], alpha=0.5)
        
        # Add labels
        labels = {n: d.get('label', n) for n, d in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title('Drug Repurposing Network', fontsize=16, fontweight='bold')
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Repurposing network visualization saved to {output_path}")
    
    def generate_repurposing_report(
        self,
        repurposing_results: Dict[str, Any],
        output_path: str
    ) -> None:
        """
        Generate comprehensive repurposing report.
        
        Args:
            repurposing_results: Results from identify_repurposing_candidates
            output_path: Path to save the report
        """
        report_lines = []
        
        # Header
        report_lines.append("DRUG REPURPOSING ANALYSIS REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Target Disease: {repurposing_results['target_disease']}")
        report_lines.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Summary
        summary = repurposing_results['summary_statistics']
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-" * 20)
        report_lines.append(f"Total candidates analyzed: {summary['total_candidates']}")
        report_lines.append(f"Top candidates selected: {summary['top_candidates_analyzed']}")
        report_lines.append(f"Repurposing potential: {summary['repurposing_potential'].upper()}")
        report_lines.append(f"Candidates above similarity threshold: {summary['similarity_statistics']['candidates_above_threshold']}")
        report_lines.append("")
        
        # Top candidates
        report_lines.append("TOP REPURPOSING CANDIDATES")
        report_lines.append("-" * 30)
        
        for i, candidate in enumerate(repurposing_results['ranked_candidates'][:10]):
            report_lines.append(f"\n{i+1}. Candidate {candidate['candidate_index']}")
            report_lines.append(f"   Similarity Score: {candidate['ranking_score']:.3f}")
            
            top_moas = candidate['moa_profile']['top_moas'][:3]
            moa_names = [moa['moa_name'] for moa in top_moas]
            report_lines.append(f"   Top MoAs: {', '.join(moa_names)}")
            
            if candidate['metadata']:
                metadata = candidate['metadata']
                if 'compound_name' in metadata:
                    report_lines.append(f"   Compound: {metadata['compound_name']}")
                if 'drug_class' in metadata:
                    report_lines.append(f"   Drug Class: {metadata['drug_class']}")
        
        # Hypotheses
        report_lines.append("\n\nREPURPOSING HYPOTHESES")
        report_lines.append("-" * 25)
        
        for i, hypothesis in enumerate(repurposing_results['hypotheses'][:5]):
            report_lines.append(f"\n{i+1}. {hypothesis['hypothesis_text']}")
            report_lines.append(f"   Confidence: {hypothesis['confidence_level'].upper()}")
            report_lines.append(f"   Shared MoAs: {', '.join(hypothesis['shared_moas'])}")
        
        # Write report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Repurposing report saved to {output_path}")
