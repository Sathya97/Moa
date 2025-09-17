"""
Therapeutic insights and clinical applications for MoA prediction models.

This module provides tools for generating therapeutic insights,
clinical decision support, and translational applications.
"""

from typing import Dict, List, Tuple, Optional, Union, Any
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from moa.utils.config import Config
from moa.utils.logger import get_logger

logger = get_logger(__name__)


class TherapeuticInsights:
    """
    Therapeutic insights generator for MoA prediction models.
    
    Provides clinical applications including:
    - Therapeutic target identification
    - Drug combination predictions
    - Biomarker discovery
    - Clinical trial design support
    - Precision medicine insights
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        moa_classes: List[str],
        clinical_data: Optional[pd.DataFrame] = None
    ):
        """
        Initialize therapeutic insights generator.
        
        Args:
            model: Trained MoA prediction model
            config: Configuration object
            moa_classes: List of MoA class names
            clinical_data: Optional clinical trial/outcome data
        """
        self.model = model
        self.config = config
        self.moa_classes = moa_classes
        self.clinical_data = clinical_data
        
        # Therapeutic parameters
        self.efficacy_threshold = config.get('applications.therapeutic.efficacy_threshold', 0.7)
        self.safety_threshold = config.get('applications.therapeutic.safety_threshold', 0.8)
        self.combination_synergy_threshold = config.get('applications.therapeutic.synergy_threshold', 0.6)
        
        self.model.eval()
        logger.info("Therapeutic insights generator initialized")
    
    def identify_therapeutic_targets(
        self,
        disease_profile: Dict[str, Any],
        compound_data_list: List[Dict[str, torch.Tensor]],
        compound_metadata: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Identify therapeutic targets for a specific disease.
        
        Args:
            disease_profile: Disease characteristics and target pathways
            compound_data_list: List of candidate compound data
            compound_metadata: Metadata for each compound
            
        Returns:
            Dictionary containing therapeutic target analysis
        """
        logger.info(f"Identifying therapeutic targets for {disease_profile.get('disease_name', 'unknown disease')}")
        
        # Get MoA predictions for all compounds
        moa_predictions = self._get_batch_moa_predictions(compound_data_list)
        
        # Analyze target pathways
        target_analysis = self._analyze_target_pathways(
            disease_profile, moa_predictions, compound_metadata
        )
        
        # Identify promising targets
        promising_targets = self._identify_promising_targets(
            target_analysis, disease_profile
        )
        
        # Assess druggability
        druggability_assessment = self._assess_target_druggability(
            promising_targets, moa_predictions, compound_metadata
        )
        
        # Generate therapeutic recommendations
        recommendations = self._generate_therapeutic_recommendations(
            promising_targets, druggability_assessment, disease_profile
        )
        
        therapeutic_results = {
            'disease_profile': disease_profile,
            'target_analysis': target_analysis,
            'promising_targets': promising_targets,
            'druggability_assessment': druggability_assessment,
            'therapeutic_recommendations': recommendations,
            'summary_metrics': self._compute_target_summary_metrics(target_analysis)
        }
        
        return therapeutic_results
    
    def predict_drug_combinations(
        self,
        compound_data_list: List[Dict[str, torch.Tensor]],
        compound_metadata: List[Dict[str, Any]],
        target_disease: str
    ) -> Dict[str, Any]:
        """
        Predict synergistic drug combinations.
        
        Args:
            compound_data_list: List of compound data
            compound_metadata: Metadata for each compound
            target_disease: Target disease for combination therapy
            
        Returns:
            Dictionary containing combination predictions
        """
        logger.info(f"Predicting drug combinations for {target_disease}")
        
        # Get MoA predictions
        moa_predictions = self._get_batch_moa_predictions(compound_data_list)
        
        # Compute pairwise combination scores
        combination_scores = self._compute_combination_scores(
            moa_predictions, compound_metadata
        )
        
        # Identify synergistic combinations
        synergistic_combinations = self._identify_synergistic_combinations(
            combination_scores, moa_predictions, compound_metadata
        )
        
        # Assess combination safety
        safety_assessment = self._assess_combination_safety(
            synergistic_combinations, compound_metadata
        )
        
        # Generate combination recommendations
        combination_recommendations = self._generate_combination_recommendations(
            synergistic_combinations, safety_assessment, target_disease
        )
        
        combination_results = {
            'target_disease': target_disease,
            'total_compounds': len(compound_data_list),
            'combination_scores': combination_scores,
            'synergistic_combinations': synergistic_combinations,
            'safety_assessment': safety_assessment,
            'recommendations': combination_recommendations,
            'combination_statistics': self._compute_combination_statistics(combination_scores)
        }
        
        return combination_results
    
    def discover_biomarkers(
        self,
        compound_data_list: List[Dict[str, torch.Tensor]],
        compound_metadata: List[Dict[str, Any]],
        clinical_outcomes: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Discover potential biomarkers for drug response.
        
        Args:
            compound_data_list: List of compound data
            compound_metadata: Metadata for each compound
            clinical_outcomes: Optional clinical outcome data
            
        Returns:
            Dictionary containing biomarker analysis
        """
        logger.info("Discovering potential biomarkers for drug response")
        
        # Get MoA predictions
        moa_predictions = self._get_batch_moa_predictions(compound_data_list)
        
        # Analyze MoA-outcome relationships
        moa_outcome_analysis = self._analyze_moa_outcome_relationships(
            moa_predictions, clinical_outcomes or []
        )
        
        # Identify predictive MoAs
        predictive_moas = self._identify_predictive_moas(
            moa_outcome_analysis, moa_predictions
        )
        
        # Generate biomarker hypotheses
        biomarker_hypotheses = self._generate_biomarker_hypotheses(
            predictive_moas, compound_metadata
        )
        
        # Assess biomarker validity
        validity_assessment = self._assess_biomarker_validity(
            biomarker_hypotheses, moa_predictions, clinical_outcomes or []
        )
        
        biomarker_results = {
            'moa_outcome_analysis': moa_outcome_analysis,
            'predictive_moas': predictive_moas,
            'biomarker_hypotheses': biomarker_hypotheses,
            'validity_assessment': validity_assessment,
            'biomarker_statistics': self._compute_biomarker_statistics(predictive_moas)
        }
        
        return biomarker_results
    
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
    
    def _analyze_target_pathways(
        self,
        disease_profile: Dict[str, Any],
        moa_predictions: List[np.ndarray],
        compound_metadata: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze target pathways for disease."""
        target_pathways = disease_profile.get('target_pathways', [])
        dysregulated_pathways = disease_profile.get('dysregulated_pathways', [])
        
        pathway_analysis = {
            'target_pathway_coverage': {},
            'compound_pathway_scores': {},
            'pathway_druggability': {}
        }
        
        # Map MoAs to pathways (simplified)
        moa_pathway_mapping = self._create_moa_pathway_mapping()
        
        # Analyze coverage of target pathways
        for pathway in target_pathways:
            relevant_moas = moa_pathway_mapping.get(pathway, [])
            if not relevant_moas:
                continue
            
            pathway_scores = []
            for pred in moa_predictions:
                pathway_score = np.mean([pred[moa_idx] for moa_idx in relevant_moas])
                pathway_scores.append(pathway_score)
            
            pathway_analysis['target_pathway_coverage'][pathway] = {
                'mean_score': np.mean(pathway_scores),
                'max_score': np.max(pathway_scores),
                'compounds_above_threshold': sum(1 for s in pathway_scores if s > self.efficacy_threshold),
                'relevant_moas': [self.moa_classes[i] for i in relevant_moas]
            }
        
        return pathway_analysis
    
    def _create_moa_pathway_mapping(self) -> Dict[str, List[int]]:
        """Create mapping from pathways to MoA indices."""
        # Simplified mapping - in practice, this would use pathway databases
        pathway_mapping = {
            'apoptosis': [i for i, moa in enumerate(self.moa_classes) if 'apoptosis' in moa.lower()],
            'cell_cycle': [i for i, moa in enumerate(self.moa_classes) if 'cycle' in moa.lower()],
            'dna_repair': [i for i, moa in enumerate(self.moa_classes) if 'dna' in moa.lower()],
            'metabolism': [i for i, moa in enumerate(self.moa_classes) if 'metabol' in moa.lower()],
            'signaling': [i for i, moa in enumerate(self.moa_classes) if 'signal' in moa.lower()]
        }
        
        return pathway_mapping
    
    def _identify_promising_targets(
        self,
        target_analysis: Dict[str, Any],
        disease_profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify promising therapeutic targets."""
        promising_targets = []
        
        target_coverage = target_analysis.get('target_pathway_coverage', {})
        
        for pathway, coverage_data in target_coverage.items():
            if coverage_data['compounds_above_threshold'] >= 3:  # Minimum druggable compounds
                target_score = (
                    coverage_data['mean_score'] * 0.4 +
                    coverage_data['max_score'] * 0.3 +
                    min(coverage_data['compounds_above_threshold'] / 10, 1.0) * 0.3
                )
                
                promising_targets.append({
                    'pathway': pathway,
                    'target_score': target_score,
                    'mean_activity': coverage_data['mean_score'],
                    'max_activity': coverage_data['max_score'],
                    'druggable_compounds': coverage_data['compounds_above_threshold'],
                    'relevant_moas': coverage_data['relevant_moas'],
                    'priority': 'high' if target_score > 0.7 else 'medium' if target_score > 0.5 else 'low'
                })
        
        return sorted(promising_targets, key=lambda x: x['target_score'], reverse=True)
    
    def _assess_target_druggability(
        self,
        promising_targets: List[Dict[str, Any]],
        moa_predictions: List[np.ndarray],
        compound_metadata: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Assess druggability of promising targets."""
        druggability_assessment = {}
        
        for target in promising_targets:
            pathway = target['pathway']
            relevant_moas = target['relevant_moas']
            
            # Find compounds with high activity against this target
            active_compounds = []
            for i, pred in enumerate(moa_predictions):
                moa_indices = [j for j, moa in enumerate(self.moa_classes) if moa in relevant_moas]
                if moa_indices:
                    pathway_score = np.mean([pred[j] for j in moa_indices])
                    if pathway_score > self.efficacy_threshold:
                        active_compounds.append({
                            'compound_index': i,
                            'pathway_score': pathway_score,
                            'metadata': compound_metadata[i]
                        })
            
            # Assess druggability factors
            druggability_score = self._compute_druggability_score(active_compounds)
            
            druggability_assessment[pathway] = {
                'druggability_score': druggability_score,
                'active_compounds': len(active_compounds),
                'chemical_diversity': self._assess_chemical_diversity(active_compounds),
                'development_stage': self._assess_development_stage(active_compounds),
                'druggability_factors': self._identify_druggability_factors(active_compounds)
            }
        
        return druggability_assessment
    
    def _compute_druggability_score(self, active_compounds: List[Dict[str, Any]]) -> float:
        """Compute druggability score for a target."""
        if not active_compounds:
            return 0.0
        
        # Factors contributing to druggability
        num_compounds = len(active_compounds)
        diversity_score = min(num_compounds / 10, 1.0)  # More compounds = higher diversity
        
        # Check for known drugs
        known_drugs = sum(1 for comp in active_compounds 
                         if comp['metadata'].get('is_approved_drug', False))
        drug_score = min(known_drugs / 3, 1.0)
        
        # Check activity scores
        activity_scores = [comp['pathway_score'] for comp in active_compounds]
        activity_score = np.mean(activity_scores)
        
        # Combined druggability score
        druggability_score = (
            diversity_score * 0.3 +
            drug_score * 0.4 +
            activity_score * 0.3
        )
        
        return druggability_score
    
    def _assess_chemical_diversity(self, active_compounds: List[Dict[str, Any]]) -> str:
        """Assess chemical diversity of active compounds."""
        if len(active_compounds) < 3:
            return 'low'
        elif len(active_compounds) < 10:
            return 'medium'
        else:
            return 'high'
    
    def _assess_development_stage(self, active_compounds: List[Dict[str, Any]]) -> Dict[str, int]:
        """Assess development stage distribution of active compounds."""
        stage_counts = {
            'approved': 0,
            'clinical': 0,
            'preclinical': 0,
            'research': 0
        }
        
        for comp in active_compounds:
            stage = comp['metadata'].get('development_stage', 'research')
            if stage in stage_counts:
                stage_counts[stage] += 1
            else:
                stage_counts['research'] += 1
        
        return stage_counts
    
    def _identify_druggability_factors(self, active_compounds: List[Dict[str, Any]]) -> List[str]:
        """Identify factors supporting druggability."""
        factors = []
        
        if len(active_compounds) >= 5:
            factors.append("Multiple active compounds identified")
        
        approved_drugs = sum(1 for comp in active_compounds 
                           if comp['metadata'].get('is_approved_drug', False))
        if approved_drugs > 0:
            factors.append(f"{approved_drugs} approved drugs target this pathway")
        
        clinical_compounds = sum(1 for comp in active_compounds 
                               if comp['metadata'].get('development_stage') == 'clinical')
        if clinical_compounds > 0:
            factors.append(f"{clinical_compounds} compounds in clinical development")
        
        high_activity = sum(1 for comp in active_compounds if comp['pathway_score'] > 0.8)
        if high_activity > 0:
            factors.append(f"{high_activity} compounds show high activity")
        
        return factors
    
    def _generate_therapeutic_recommendations(
        self,
        promising_targets: List[Dict[str, Any]],
        druggability_assessment: Dict[str, Any],
        disease_profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate therapeutic recommendations."""
        recommendations = []
        
        for target in promising_targets[:5]:  # Top 5 targets
            pathway = target['pathway']
            druggability = druggability_assessment.get(pathway, {})
            
            recommendation = {
                'target_pathway': pathway,
                'recommendation_type': self._determine_recommendation_type(target, druggability),
                'priority': target['priority'],
                'rationale': self._generate_recommendation_rationale(target, druggability),
                'next_steps': self._suggest_next_steps(target, druggability),
                'timeline': self._estimate_timeline(target, druggability),
                'success_probability': self._estimate_success_probability(target, druggability)
            }
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _determine_recommendation_type(
        self,
        target: Dict[str, Any],
        druggability: Dict[str, Any]
    ) -> str:
        """Determine type of therapeutic recommendation."""
        druggability_score = druggability.get('druggability_score', 0)
        target_score = target['target_score']
        
        if druggability_score > 0.7 and target_score > 0.7:
            return 'immediate_development'
        elif druggability_score > 0.5 and target_score > 0.6:
            return 'lead_optimization'
        elif target_score > 0.6:
            return 'target_validation'
        else:
            return 'basic_research'
    
    def _generate_recommendation_rationale(
        self,
        target: Dict[str, Any],
        druggability: Dict[str, Any]
    ) -> str:
        """Generate rationale for therapeutic recommendation."""
        pathway = target['pathway']
        target_score = target['target_score']
        druggability_score = druggability.get('druggability_score', 0)
        
        rationale = (
            f"The {pathway} pathway shows strong therapeutic potential "
            f"(target score: {target_score:.2f}) with "
            f"{'high' if druggability_score > 0.7 else 'moderate' if druggability_score > 0.5 else 'limited'} "
            f"druggability (score: {druggability_score:.2f}). "
        )
        
        active_compounds = druggability.get('active_compounds', 0)
        if active_compounds > 0:
            rationale += f"{active_compounds} compounds show activity against this target. "
        
        factors = druggability.get('druggability_factors', [])
        if factors:
            rationale += "Supporting factors: " + "; ".join(factors[:2]) + "."
        
        return rationale
    
    def _suggest_next_steps(
        self,
        target: Dict[str, Any],
        druggability: Dict[str, Any]
    ) -> List[str]:
        """Suggest next steps for target development."""
        recommendation_type = self._determine_recommendation_type(target, druggability)
        
        if recommendation_type == 'immediate_development':
            return [
                "Initiate lead compound optimization",
                "Conduct in vivo efficacy studies",
                "Prepare IND application"
            ]
        elif recommendation_type == 'lead_optimization':
            return [
                "Screen additional compound libraries",
                "Optimize ADMET properties",
                "Validate target engagement"
            ]
        elif recommendation_type == 'target_validation':
            return [
                "Conduct target validation studies",
                "Develop target-specific assays",
                "Identify biomarkers"
            ]
        else:
            return [
                "Investigate target biology",
                "Develop screening assays",
                "Search for tool compounds"
            ]
    
    def _estimate_timeline(
        self,
        target: Dict[str, Any],
        druggability: Dict[str, Any]
    ) -> str:
        """Estimate development timeline."""
        recommendation_type = self._determine_recommendation_type(target, druggability)
        
        timeline_map = {
            'immediate_development': '2-4 years to clinical trials',
            'lead_optimization': '3-5 years to clinical trials',
            'target_validation': '4-6 years to clinical trials',
            'basic_research': '5-8 years to clinical trials'
        }
        
        return timeline_map.get(recommendation_type, 'Timeline uncertain')
    
    def _estimate_success_probability(
        self,
        target: Dict[str, Any],
        druggability: Dict[str, Any]
    ) -> str:
        """Estimate probability of success."""
        target_score = target['target_score']
        druggability_score = druggability.get('druggability_score', 0)
        
        combined_score = (target_score + druggability_score) / 2
        
        if combined_score > 0.7:
            return 'high (>60%)'
        elif combined_score > 0.5:
            return 'medium (30-60%)'
        else:
            return 'low (<30%)'
    
    def _compute_target_summary_metrics(
        self,
        target_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute summary metrics for target analysis."""
        target_coverage = target_analysis.get('target_pathway_coverage', {})
        
        if not target_coverage:
            return {}
        
        mean_scores = [data['mean_score'] for data in target_coverage.values()]
        max_scores = [data['max_score'] for data in target_coverage.values()]
        compound_counts = [data['compounds_above_threshold'] for data in target_coverage.values()]
        
        summary_metrics = {
            'total_pathways_analyzed': len(target_coverage),
            'mean_pathway_activity': np.mean(mean_scores),
            'max_pathway_activity': np.max(max_scores),
            'total_druggable_compounds': sum(compound_counts),
            'pathways_with_compounds': sum(1 for count in compound_counts if count > 0),
            'high_activity_pathways': sum(1 for score in mean_scores if score > self.efficacy_threshold)
        }
        
        return summary_metrics
    
    def _compute_combination_scores(
        self,
        moa_predictions: List[np.ndarray],
        compound_metadata: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Compute pairwise combination scores."""
        combination_scores = []
        
        for i in range(len(moa_predictions)):
            for j in range(i + 1, len(moa_predictions)):
                pred_i = moa_predictions[i]
                pred_j = moa_predictions[j]
                
                # Compute complementarity score
                complementarity = self._compute_complementarity(pred_i, pred_j)
                
                # Compute synergy potential
                synergy_potential = self._compute_synergy_potential(pred_i, pred_j)
                
                # Combined score
                combination_score = (complementarity + synergy_potential) / 2
                
                combination_scores.append({
                    'compound_1_index': i,
                    'compound_2_index': j,
                    'compound_1_name': compound_metadata[i].get('compound_name', f'Compound_{i}'),
                    'compound_2_name': compound_metadata[j].get('compound_name', f'Compound_{j}'),
                    'complementarity_score': complementarity,
                    'synergy_potential': synergy_potential,
                    'combination_score': combination_score
                })
        
        return sorted(combination_scores, key=lambda x: x['combination_score'], reverse=True)
    
    def _compute_complementarity(self, pred_i: np.ndarray, pred_j: np.ndarray) -> float:
        """Compute complementarity between two MoA profiles."""
        # Complementarity: compounds target different pathways
        overlap = np.minimum(pred_i, pred_j)
        complementarity = 1 - np.mean(overlap)
        return complementarity
    
    def _compute_synergy_potential(self, pred_i: np.ndarray, pred_j: np.ndarray) -> float:
        """Compute synergy potential between two MoA profiles."""
        # Synergy: combined effect greater than individual effects
        individual_max = np.maximum(pred_i, pred_j)
        combined_effect = np.minimum(pred_i + pred_j, 1.0)  # Cap at 1.0
        synergy = np.mean(combined_effect - individual_max)
        return max(synergy, 0)  # Only positive synergy
    
    def _identify_synergistic_combinations(
        self,
        combination_scores: List[Dict[str, Any]],
        moa_predictions: List[np.ndarray],
        compound_metadata: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify synergistic drug combinations."""
        synergistic_combinations = []
        
        for combo in combination_scores:
            if combo['combination_score'] > self.combination_synergy_threshold:
                # Add additional analysis
                i, j = combo['compound_1_index'], combo['compound_2_index']
                
                # Identify shared and unique MoAs
                pred_i, pred_j = moa_predictions[i], moa_predictions[j]
                shared_moas = []
                unique_i_moas = []
                unique_j_moas = []
                
                for moa_idx, (score_i, score_j) in enumerate(zip(pred_i, pred_j)):
                    if score_i > 0.5 and score_j > 0.5:
                        shared_moas.append(self.moa_classes[moa_idx])
                    elif score_i > 0.5:
                        unique_i_moas.append(self.moa_classes[moa_idx])
                    elif score_j > 0.5:
                        unique_j_moas.append(self.moa_classes[moa_idx])
                
                combo_analysis = combo.copy()
                combo_analysis.update({
                    'shared_moas': shared_moas,
                    'unique_moas_compound_1': unique_i_moas,
                    'unique_moas_compound_2': unique_j_moas,
                    'synergy_mechanism': self._predict_synergy_mechanism(shared_moas, unique_i_moas, unique_j_moas)
                })
                
                synergistic_combinations.append(combo_analysis)
        
        return synergistic_combinations
    
    def _predict_synergy_mechanism(
        self,
        shared_moas: List[str],
        unique_i_moas: List[str],
        unique_j_moas: List[str]
    ) -> str:
        """Predict mechanism of synergy."""
        if len(shared_moas) > 2:
            return 'additive_effect'
        elif len(unique_i_moas) > 0 and len(unique_j_moas) > 0:
            return 'complementary_pathways'
        elif len(shared_moas) > 0 and (len(unique_i_moas) > 0 or len(unique_j_moas) > 0):
            return 'pathway_enhancement'
        else:
            return 'unknown_mechanism'
    
    def _assess_combination_safety(
        self,
        synergistic_combinations: List[Dict[str, Any]],
        compound_metadata: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Assess safety of drug combinations."""
        safety_assessment = {}
        
        for combo in synergistic_combinations:
            combo_id = f"{combo['compound_1_index']}_{combo['compound_2_index']}"
            
            # Get safety profiles
            safety_1 = compound_metadata[combo['compound_1_index']].get('safety_profile', {})
            safety_2 = compound_metadata[combo['compound_2_index']].get('safety_profile', {})
            
            # Assess interaction risk
            interaction_risk = self._assess_interaction_risk(safety_1, safety_2, combo['shared_moas'])
            
            safety_assessment[combo_id] = {
                'interaction_risk': interaction_risk,
                'safety_score': self._compute_combination_safety_score(safety_1, safety_2, interaction_risk),
                'contraindications': self._identify_contraindications(safety_1, safety_2),
                'monitoring_requirements': self._suggest_monitoring(safety_1, safety_2, combo['shared_moas'])
            }
        
        return safety_assessment
    
    def _assess_interaction_risk(
        self,
        safety_1: Dict[str, Any],
        safety_2: Dict[str, Any],
        shared_moas: List[str]
    ) -> str:
        """Assess drug-drug interaction risk."""
        # Simplified risk assessment
        if len(shared_moas) > 3:
            return 'high'
        elif len(shared_moas) > 1:
            return 'medium'
        else:
            return 'low'
    
    def _compute_combination_safety_score(
        self,
        safety_1: Dict[str, Any],
        safety_2: Dict[str, Any],
        interaction_risk: str
    ) -> float:
        """Compute overall safety score for combination."""
        # Simplified safety scoring
        base_safety = 0.8  # Assume reasonable safety for individual compounds
        
        risk_penalties = {'high': 0.4, 'medium': 0.2, 'low': 0.1}
        penalty = risk_penalties.get(interaction_risk, 0.2)
        
        safety_score = base_safety - penalty
        return max(safety_score, 0.0)
    
    def _identify_contraindications(
        self,
        safety_1: Dict[str, Any],
        safety_2: Dict[str, Any]
    ) -> List[str]:
        """Identify contraindications for combination."""
        contraindications = []
        
        # Check for overlapping toxicities
        toxicities_1 = set(safety_1.get('known_toxicities', []))
        toxicities_2 = set(safety_2.get('known_toxicities', []))
        
        overlapping_toxicities = toxicities_1 & toxicities_2
        if overlapping_toxicities:
            contraindications.append(f"Overlapping toxicities: {', '.join(overlapping_toxicities)}")
        
        return contraindications
    
    def _suggest_monitoring(
        self,
        safety_1: Dict[str, Any],
        safety_2: Dict[str, Any],
        shared_moas: List[str]
    ) -> List[str]:
        """Suggest monitoring requirements for combination."""
        monitoring = []
        
        if len(shared_moas) > 2:
            monitoring.append("Enhanced pharmacokinetic monitoring")
            monitoring.append("Frequent efficacy assessments")
        
        if 'hepatotoxicity' in safety_1.get('known_toxicities', []) or 'hepatotoxicity' in safety_2.get('known_toxicities', []):
            monitoring.append("Liver function monitoring")
        
        return monitoring
    
    def _generate_combination_recommendations(
        self,
        synergistic_combinations: List[Dict[str, Any]],
        safety_assessment: Dict[str, Any],
        target_disease: str
    ) -> List[Dict[str, Any]]:
        """Generate combination therapy recommendations."""
        recommendations = []
        
        for combo in synergistic_combinations[:10]:  # Top 10 combinations
            combo_id = f"{combo['compound_1_index']}_{combo['compound_2_index']}"
            safety_data = safety_assessment.get(combo_id, {})
            
            recommendation = {
                'combination': f"{combo['compound_1_name']} + {combo['compound_2_name']}",
                'synergy_score': combo['combination_score'],
                'safety_score': safety_data.get('safety_score', 0.5),
                'recommendation_strength': self._determine_combination_strength(combo, safety_data),
                'rationale': self._generate_combination_rationale(combo, safety_data),
                'clinical_considerations': self._suggest_clinical_considerations(combo, safety_data),
                'development_priority': self._assess_development_priority(combo, safety_data)
            }
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _determine_combination_strength(
        self,
        combo: Dict[str, Any],
        safety_data: Dict[str, Any]
    ) -> str:
        """Determine strength of combination recommendation."""
        synergy_score = combo['combination_score']
        safety_score = safety_data.get('safety_score', 0.5)
        
        combined_score = (synergy_score + safety_score) / 2
        
        if combined_score > 0.7:
            return 'strong'
        elif combined_score > 0.5:
            return 'moderate'
        else:
            return 'weak'
    
    def _generate_combination_rationale(
        self,
        combo: Dict[str, Any],
        safety_data: Dict[str, Any]
    ) -> str:
        """Generate rationale for combination recommendation."""
        mechanism = combo.get('synergy_mechanism', 'unknown')
        synergy_score = combo['combination_score']
        
        rationale = (
            f"This combination shows {mechanism.replace('_', ' ')} "
            f"with a synergy score of {synergy_score:.2f}. "
        )
        
        if combo.get('shared_moas'):
            rationale += f"Compounds share {len(combo['shared_moas'])} mechanisms. "
        
        if combo.get('unique_moas_compound_1') and combo.get('unique_moas_compound_2'):
            rationale += "Complementary mechanisms provide comprehensive pathway coverage."
        
        return rationale
    
    def _suggest_clinical_considerations(
        self,
        combo: Dict[str, Any],
        safety_data: Dict[str, Any]
    ) -> List[str]:
        """Suggest clinical considerations for combination."""
        considerations = []
        
        interaction_risk = safety_data.get('interaction_risk', 'medium')
        if interaction_risk == 'high':
            considerations.append("Careful dose titration required")
            considerations.append("Enhanced safety monitoring")
        
        if combo.get('synergy_mechanism') == 'additive_effect':
            considerations.append("Consider dose reduction for individual components")
        
        monitoring = safety_data.get('monitoring_requirements', [])
        if monitoring:
            considerations.extend(monitoring)
        
        return considerations
    
    def _assess_development_priority(
        self,
        combo: Dict[str, Any],
        safety_data: Dict[str, Any]
    ) -> str:
        """Assess development priority for combination."""
        synergy_score = combo['combination_score']
        safety_score = safety_data.get('safety_score', 0.5)
        
        if synergy_score > 0.8 and safety_score > 0.7:
            return 'high'
        elif synergy_score > 0.6 and safety_score > 0.5:
            return 'medium'
        else:
            return 'low'
    
    def _compute_combination_statistics(
        self,
        combination_scores: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute statistics for combination analysis."""
        if not combination_scores:
            return {}
        
        scores = [combo['combination_score'] for combo in combination_scores]
        
        statistics = {
            'total_combinations': len(combination_scores),
            'mean_combination_score': np.mean(scores),
            'max_combination_score': np.max(scores),
            'synergistic_combinations': sum(1 for score in scores if score > self.combination_synergy_threshold),
            'high_synergy_combinations': sum(1 for score in scores if score > 0.8),
            'score_distribution': {
                'high (>0.8)': sum(1 for score in scores if score > 0.8),
                'medium (0.6-0.8)': sum(1 for score in scores if 0.6 <= score <= 0.8),
                'low (<0.6)': sum(1 for score in scores if score < 0.6)
            }
        }
        
        return statistics
    
    def _analyze_moa_outcome_relationships(
        self,
        moa_predictions: List[np.ndarray],
        clinical_outcomes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze relationships between MoAs and clinical outcomes."""
        # Simplified analysis - would require real clinical data
        moa_outcome_analysis = {
            'moa_efficacy_correlations': {},
            'outcome_predictive_moas': {},
            'moa_safety_associations': {}
        }
        
        if not clinical_outcomes:
            logger.warning("No clinical outcomes provided for analysis")
            return moa_outcome_analysis
        
        # This would implement real statistical analysis with clinical data
        return moa_outcome_analysis
    
    def _identify_predictive_moas(
        self,
        moa_outcome_analysis: Dict[str, Any],
        moa_predictions: List[np.ndarray]
    ) -> List[Dict[str, Any]]:
        """Identify MoAs predictive of clinical outcomes."""
        # Simplified implementation
        predictive_moas = []
        
        # In practice, this would use statistical analysis of clinical data
        for i, moa_name in enumerate(self.moa_classes):
            # Simulate predictive value
            predictive_value = np.random.random()
            
            if predictive_value > 0.7:
                predictive_moas.append({
                    'moa_name': moa_name,
                    'moa_index': i,
                    'predictive_value': predictive_value,
                    'outcome_type': 'efficacy',
                    'confidence': 'high' if predictive_value > 0.8 else 'medium'
                })
        
        return sorted(predictive_moas, key=lambda x: x['predictive_value'], reverse=True)
    
    def _generate_biomarker_hypotheses(
        self,
        predictive_moas: List[Dict[str, Any]],
        compound_metadata: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate biomarker hypotheses."""
        biomarker_hypotheses = []
        
        for moa in predictive_moas[:10]:  # Top 10 predictive MoAs
            hypothesis = {
                'biomarker_type': 'pathway_activity',
                'moa_target': moa['moa_name'],
                'hypothesis_text': f"Activity against {moa['moa_name']} pathway may predict clinical response",
                'biomarker_measurement': f"Expression/activity of {moa['moa_name']} pathway components",
                'clinical_utility': self._assess_biomarker_utility(moa),
                'validation_requirements': self._suggest_biomarker_validation(moa)
            }
            
            biomarker_hypotheses.append(hypothesis)
        
        return biomarker_hypotheses
    
    def _assess_biomarker_utility(self, moa: Dict[str, Any]) -> str:
        """Assess clinical utility of biomarker."""
        predictive_value = moa['predictive_value']
        
        if predictive_value > 0.8:
            return 'high_utility'
        elif predictive_value > 0.6:
            return 'medium_utility'
        else:
            return 'low_utility'
    
    def _suggest_biomarker_validation(self, moa: Dict[str, Any]) -> List[str]:
        """Suggest validation steps for biomarker."""
        return [
            f"Develop assay for {moa['moa_name']} pathway activity",
            "Validate in retrospective clinical samples",
            "Test in prospective clinical trial",
            "Establish clinical cutoff values"
        ]
    
    def _assess_biomarker_validity(
        self,
        biomarker_hypotheses: List[Dict[str, Any]],
        moa_predictions: List[np.ndarray],
        clinical_outcomes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Assess validity of biomarker hypotheses."""
        # Simplified validity assessment
        validity_assessment = {
            'high_validity_biomarkers': [],
            'medium_validity_biomarkers': [],
            'low_validity_biomarkers': [],
            'validation_statistics': {}
        }
        
        for hypothesis in biomarker_hypotheses:
            utility = hypothesis['clinical_utility']
            
            if utility == 'high_utility':
                validity_assessment['high_validity_biomarkers'].append(hypothesis)
            elif utility == 'medium_utility':
                validity_assessment['medium_validity_biomarkers'].append(hypothesis)
            else:
                validity_assessment['low_validity_biomarkers'].append(hypothesis)
        
        return validity_assessment
    
    def _compute_biomarker_statistics(
        self,
        predictive_moas: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute statistics for biomarker analysis."""
        if not predictive_moas:
            return {}
        
        predictive_values = [moa['predictive_value'] for moa in predictive_moas]
        
        statistics = {
            'total_predictive_moas': len(predictive_moas),
            'mean_predictive_value': np.mean(predictive_values),
            'max_predictive_value': np.max(predictive_values),
            'high_predictive_moas': sum(1 for val in predictive_values if val > 0.8),
            'medium_predictive_moas': sum(1 for val in predictive_values if 0.6 <= val <= 0.8),
            'efficacy_biomarkers': sum(1 for moa in predictive_moas if moa['outcome_type'] == 'efficacy'),
            'safety_biomarkers': sum(1 for moa in predictive_moas if moa['outcome_type'] == 'safety')
        }
        
        return statistics
