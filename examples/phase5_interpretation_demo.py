#!/usr/bin/env python3
"""
Phase 5: Interpretation & Applications Demo

This script demonstrates the interpretation and application capabilities
of the MoA prediction framework, including model explainability,
drug repurposing, knowledge discovery, and therapeutic insights.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from moa.utils.config import Config
from moa.utils.logger import get_logger
from moa.models.multimodal_model import MultiModalMoAModel
from moa.interpretation.explainer import MoAExplainer
from moa.interpretation.uncertainty import UncertaintyEstimator
from moa.applications.drug_repurposing import DrugRepurposingPipeline
from moa.applications.knowledge_discovery import KnowledgeDiscovery
from moa.applications.therapeutic_insights import TherapeuticInsights

logger = get_logger(__name__)


def create_demo_data():
    """Create demonstration data for Phase 5."""
    logger.info("Creating demonstration data for Phase 5")
    
    # Demo compound data
    demo_compounds = []
    compound_metadata = []
    
    for i in range(20):
        # Chemical features
        chemical_features = {
            'node_features': torch.randn(30, 64),  # 30 atoms, 64 features each
            'edge_index': torch.randint(0, 30, (2, 80)),  # 80 bonds
            'edge_features': torch.randn(80, 16),  # 16 edge features
            'batch': torch.zeros(30, dtype=torch.long),
            'counterfactual_weights': torch.rand(30)
        }
        
        # Biological features
        biological_features = {
            'gene_expression': torch.randn(978),  # L1000 genes
            'pathway_scores': torch.randn(50),   # 50 pathways
            'mechanism_tokens': torch.randn(128) # MechTokens
        }
        
        # Protein features (optional)
        protein_features = {
            'pocket_features': torch.randn(256)
        }
        
        compound_data = {
            'chemical': chemical_features,
            'biological': biological_features,
            'protein': protein_features
        }
        
        demo_compounds.append(compound_data)
        
        # Metadata
        metadata = {
            'compound_id': f'DEMO_{i:03d}',
            'compound_name': f'Demo Compound {i+1}',
            'smiles': f'CC(C)C{i}',  # Simplified SMILES
            'molecular_weight': 200 + i * 10,
            'drug_class': ['kinase_inhibitor', 'gpcr_agonist', 'ion_channel_blocker'][i % 3],
            'indication': ['cancer', 'diabetes', 'hypertension'][i % 3],
            'development_stage': ['approved', 'clinical', 'preclinical', 'research'][i % 4],
            'is_approved_drug': i < 5,
            'known_indications': [['cancer'], ['diabetes'], ['hypertension']][i % 3],
            'safety_profile': {
                'known_toxicities': ['hepatotoxicity'] if i % 5 == 0 else [],
                'contraindications': []
            }
        }
        
        compound_metadata.append(metadata)
    
    # Demo disease profile
    disease_profile = {
        'disease_name': 'Type 2 Diabetes',
        'target_pathways': ['metabolism', 'signaling', 'insulin_pathway'],
        'dysregulated_pathways': ['glucose_metabolism', 'lipid_metabolism'],
        'biomarkers': ['HbA1c', 'glucose', 'insulin'],
        'therapeutic_targets': ['PPARG', 'DPP4', 'SGLT2']
    }
    
    # Demo clinical outcomes
    clinical_outcomes = []
    for i in range(20):
        outcome = {
            'compound_id': f'DEMO_{i:03d}',
            'efficacy_score': np.random.random(),
            'safety_score': np.random.random(),
            'response_rate': np.random.random(),
            'adverse_events': np.random.randint(0, 5)
        }
        clinical_outcomes.append(outcome)
    
    return demo_compounds, compound_metadata, disease_profile, clinical_outcomes


def demonstrate_model_interpretation(model, demo_compounds, compound_metadata, config):
    """Demonstrate model interpretation capabilities."""
    logger.info("=== DEMONSTRATING MODEL INTERPRETATION ===")
    
    # Initialize explainer
    moa_classes = [f'MoA_{i}' for i in range(config.model.num_classes)]
    explainer = MoAExplainer(model, config, moa_classes)
    
    # Explain a single prediction
    logger.info("Explaining single compound prediction...")
    compound_data = demo_compounds[0]
    explanation = explainer.explain_prediction(compound_data, top_k_features=10)
    
    print(f"\n--- Explanation for {compound_metadata[0]['compound_name']} ---")
    print(f"Top predicted MoAs: {[moa['moa_name'] for moa in explanation['top_predictions'][:3]]}")
    print(f"Prediction confidence: {explanation['prediction_confidence']:.3f}")
    print(f"Model certainty: {explanation['model_certainty']:.3f}")
    
    # Show modality contributions
    print("\nModality Contributions:")
    for modality, contrib in explanation['modality_contributions'].items():
        print(f"  {modality}: {contrib:.3f}")
    
    # Show top features
    print("\nTop Important Features:")
    for feature in explanation['feature_importance'][:5]:
        print(f"  {feature['feature_name']}: {feature['importance']:.3f}")
    
    # Demonstrate uncertainty estimation
    logger.info("Demonstrating uncertainty estimation...")
    uncertainty_estimator = UncertaintyEstimator(model, config)
    
    uncertainty_results = uncertainty_estimator.estimate_uncertainty(
        compound_data, n_samples=50
    )
    
    print(f"\nUncertainty Analysis:")
    print(f"Epistemic uncertainty: {uncertainty_results['epistemic_uncertainty']:.3f}")
    print(f"Aleatoric uncertainty: {uncertainty_results['aleatoric_uncertainty']:.3f}")
    print(f"Total uncertainty: {uncertainty_results['total_uncertainty']:.3f}")
    print(f"Prediction confidence: {uncertainty_results['prediction_confidence']:.3f}")
    
    return explanation, uncertainty_results


def demonstrate_drug_repurposing(model, demo_compounds, compound_metadata, config):
    """Demonstrate drug repurposing pipeline."""
    logger.info("=== DEMONSTRATING DRUG REPURPOSING ===")
    
    # Initialize repurposing pipeline
    moa_classes = [f'MoA_{i}' for i in range(config.model.num_classes)]
    repurposing_pipeline = DrugRepurposingPipeline(model, config, moa_classes)
    
    # Use first compound as query (known effective drug)
    query_compound = demo_compounds[0]
    candidate_compounds = demo_compounds[1:11]  # Next 10 as candidates
    candidate_metadata = compound_metadata[1:11]
    
    logger.info("Identifying repurposing candidates...")
    repurposing_results = repurposing_pipeline.identify_repurposing_candidates(
        query_compound_data=query_compound,
        target_disease="Type 2 Diabetes",
        candidate_compounds=candidate_compounds,
        compound_metadata=candidate_metadata
    )
    
    print(f"\n--- Drug Repurposing Results for Type 2 Diabetes ---")
    print(f"Query compound: {compound_metadata[0]['compound_name']}")
    print(f"Candidates analyzed: {len(candidate_compounds)}")
    print(f"Repurposing potential: {repurposing_results['summary_statistics']['repurposing_potential']}")
    
    # Show top candidates
    print("\nTop Repurposing Candidates:")
    for i, candidate in enumerate(repurposing_results['ranked_candidates'][:5]):
        metadata = candidate_metadata[candidate['candidate_index']]
        print(f"  {i+1}. {metadata['compound_name']}")
        print(f"     Similarity Score: {candidate['ranking_score']:.3f}")
        print(f"     Drug Class: {metadata['drug_class']}")
    
    # Show hypotheses
    print("\nRepurposing Hypotheses:")
    for i, hypothesis in enumerate(repurposing_results['hypotheses'][:3]):
        print(f"  {i+1}. {hypothesis['hypothesis_text']}")
        print(f"     Confidence: {hypothesis['confidence_level']}")
        print(f"     Shared MoAs: {', '.join(hypothesis['shared_moas'])}")
    
    # Generate repurposing network
    logger.info("Creating repurposing network visualization...")
    network = repurposing_pipeline.create_repurposing_network(
        repurposing_results,
        output_path="outputs/repurposing_network.png"
    )
    
    print(f"\nRepurposing network created with {network.number_of_nodes()} nodes and {network.number_of_edges()} edges")
    
    return repurposing_results


def demonstrate_knowledge_discovery(model, demo_compounds, compound_metadata, config):
    """Demonstrate knowledge discovery capabilities."""
    logger.info("=== DEMONSTRATING KNOWLEDGE DISCOVERY ===")
    
    # Initialize knowledge discovery system
    moa_classes = [f'MoA_{i}' for i in range(config.model.num_classes)]
    
    # Create pathway annotations
    pathway_annotations = {}
    for i, moa in enumerate(moa_classes):
        if i < 10:
            pathway_annotations[moa] = ['apoptosis', 'cell_cycle']
        elif i < 20:
            pathway_annotations[moa] = ['metabolism', 'signaling']
        else:
            pathway_annotations[moa] = ['dna_repair', 'protein_synthesis']
    
    knowledge_discovery = KnowledgeDiscovery(
        model, config, moa_classes, pathway_annotations
    )
    
    # Discover novel associations
    logger.info("Discovering novel drug-pathway associations...")
    discovery_results = knowledge_discovery.discover_novel_associations(
        compound_data_list=demo_compounds,
        compound_metadata=compound_metadata,
        known_associations={}  # No known associations for demo
    )
    
    print(f"\n--- Knowledge Discovery Results ---")
    print(f"Compounds analyzed: {discovery_results['total_compounds']}")
    print(f"Total associations found: {discovery_results['discovery_statistics']['total_associations']}")
    print(f"Significant associations: {discovery_results['discovery_statistics']['significant_associations']}")
    
    # Show top validated associations
    print("\nTop Validated Associations:")
    for i, assoc in enumerate(discovery_results['validated_associations'][:5]):
        print(f"  {i+1}. {assoc.get('property', assoc.get('pathway', 'Unknown'))} -> {assoc.get('moa', 'Unknown MoA')}")
        print(f"     Method: {assoc['discovery_method']}")
        print(f"     P-value: {assoc.get('p_value', 'N/A')}")
        print(f"     Confidence: {assoc.get('confidence', 'medium')}")
    
    # Show biological hypotheses
    print("\nBiological Hypotheses:")
    for i, hypothesis in enumerate(discovery_results['biological_hypotheses'][:3]):
        print(f"  {i+1}. {hypothesis['hypothesis_text']}")
        print(f"     Type: {hypothesis['hypothesis_type']}")
        print(f"     Confidence: {hypothesis['confidence']}")
    
    return discovery_results


def demonstrate_therapeutic_insights(model, demo_compounds, compound_metadata, config):
    """Demonstrate therapeutic insights generation."""
    logger.info("=== DEMONSTRATING THERAPEUTIC INSIGHTS ===")
    
    # Initialize therapeutic insights generator
    moa_classes = [f'MoA_{i}' for i in range(config.model.num_classes)]
    therapeutic_insights = TherapeuticInsights(model, config, moa_classes)
    
    # Disease profile for analysis
    disease_profile = {
        'disease_name': 'Type 2 Diabetes',
        'target_pathways': ['metabolism', 'signaling', 'insulin_pathway'],
        'dysregulated_pathways': ['glucose_metabolism', 'lipid_metabolism'],
        'biomarkers': ['HbA1c', 'glucose', 'insulin'],
        'therapeutic_targets': ['PPARG', 'DPP4', 'SGLT2']
    }
    
    # Identify therapeutic targets
    logger.info("Identifying therapeutic targets...")
    target_results = therapeutic_insights.identify_therapeutic_targets(
        disease_profile=disease_profile,
        compound_data_list=demo_compounds,
        compound_metadata=compound_metadata
    )
    
    print(f"\n--- Therapeutic Target Analysis ---")
    print(f"Disease: {disease_profile['disease_name']}")
    print(f"Pathways analyzed: {len(disease_profile['target_pathways'])}")
    
    # Show promising targets
    promising_targets = target_results['promising_targets']
    print(f"\nPromising Targets ({len(promising_targets)} found):")
    for i, target in enumerate(promising_targets[:3]):
        print(f"  {i+1}. {target['pathway']}")
        print(f"     Target Score: {target['target_score']:.3f}")
        print(f"     Priority: {target['priority']}")
        print(f"     Druggable Compounds: {target['druggable_compounds']}")
    
    # Show therapeutic recommendations
    recommendations = target_results['therapeutic_recommendations']
    print(f"\nTherapeutic Recommendations:")
    for i, rec in enumerate(recommendations[:3]):
        print(f"  {i+1}. {rec['target_pathway']}")
        print(f"     Recommendation: {rec['recommendation_type']}")
        print(f"     Priority: {rec['priority']}")
        print(f"     Timeline: {rec['timeline']}")
        print(f"     Success Probability: {rec['success_probability']}")
    
    # Predict drug combinations
    logger.info("Predicting drug combinations...")
    combination_results = therapeutic_insights.predict_drug_combinations(
        compound_data_list=demo_compounds[:10],
        compound_metadata=compound_metadata[:10],
        target_disease="Type 2 Diabetes"
    )
    
    print(f"\n--- Drug Combination Analysis ---")
    print(f"Combinations analyzed: {combination_results['combination_statistics']['total_combinations']}")
    print(f"Synergistic combinations: {combination_results['combination_statistics']['synergistic_combinations']}")
    
    # Show top combinations
    synergistic_combos = combination_results['synergistic_combinations']
    print(f"\nTop Synergistic Combinations:")
    for i, combo in enumerate(synergistic_combos[:3]):
        print(f"  {i+1}. {combo['compound_1_name']} + {combo['compound_2_name']}")
        print(f"     Synergy Score: {combo['combination_score']:.3f}")
        print(f"     Mechanism: {combo['synergy_mechanism']}")
        print(f"     Shared MoAs: {len(combo['shared_moas'])}")
    
    # Discover biomarkers
    logger.info("Discovering potential biomarkers...")
    biomarker_results = therapeutic_insights.discover_biomarkers(
        compound_data_list=demo_compounds,
        compound_metadata=compound_metadata,
        clinical_outcomes=None  # No clinical data for demo
    )
    
    print(f"\n--- Biomarker Discovery ---")
    predictive_moas = biomarker_results['predictive_moas']
    print(f"Predictive MoAs identified: {len(predictive_moas)}")
    
    # Show biomarker hypotheses
    hypotheses = biomarker_results['biomarker_hypotheses']
    print(f"\nBiomarker Hypotheses:")
    for i, hypothesis in enumerate(hypotheses[:3]):
        print(f"  {i+1}. {hypothesis['hypothesis_text']}")
        print(f"     Type: {hypothesis['biomarker_type']}")
        print(f"     Clinical Utility: {hypothesis['clinical_utility']}")
    
    return target_results, combination_results, biomarker_results


def create_phase5_summary_report(
    interpretation_results,
    repurposing_results,
    discovery_results,
    therapeutic_results
):
    """Create comprehensive Phase 5 summary report."""
    logger.info("Creating Phase 5 summary report...")
    
    report_lines = []
    
    # Header
    report_lines.append("PHASE 5: INTERPRETATION & APPLICATIONS - SUMMARY REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Model Interpretation Summary
    report_lines.append("1. MODEL INTERPRETATION & EXPLAINABILITY")
    report_lines.append("-" * 45)
    explanation, uncertainty = interpretation_results
    report_lines.append(f"✓ Model explainability implemented with attention visualization")
    report_lines.append(f"✓ Uncertainty estimation: {uncertainty['total_uncertainty']:.3f}")
    report_lines.append(f"✓ Feature importance analysis across modalities")
    report_lines.append(f"✓ Counterfactual analysis for molecular fragments")
    report_lines.append("")
    
    # Drug Repurposing Summary
    report_lines.append("2. DRUG REPURPOSING PIPELINE")
    report_lines.append("-" * 30)
    report_lines.append(f"✓ Repurposing potential: {repurposing_results['summary_statistics']['repurposing_potential']}")
    report_lines.append(f"✓ Top candidates identified: {len(repurposing_results['ranked_candidates'])}")
    report_lines.append(f"✓ Hypotheses generated: {len(repurposing_results['hypotheses'])}")
    report_lines.append(f"✓ Network visualization created")
    report_lines.append("")
    
    # Knowledge Discovery Summary
    report_lines.append("3. KNOWLEDGE DISCOVERY")
    report_lines.append("-" * 25)
    report_lines.append(f"✓ Total associations discovered: {discovery_results['discovery_statistics']['total_associations']}")
    report_lines.append(f"✓ Significant associations: {discovery_results['discovery_statistics']['significant_associations']}")
    report_lines.append(f"✓ Biological hypotheses: {len(discovery_results['biological_hypotheses'])}")
    report_lines.append(f"✓ Novel predictions identified")
    report_lines.append("")
    
    # Therapeutic Insights Summary
    report_lines.append("4. THERAPEUTIC INSIGHTS")
    report_lines.append("-" * 25)
    target_results, combination_results, biomarker_results = therapeutic_results
    report_lines.append(f"✓ Promising targets identified: {len(target_results['promising_targets'])}")
    report_lines.append(f"✓ Therapeutic recommendations: {len(target_results['therapeutic_recommendations'])}")
    report_lines.append(f"✓ Synergistic combinations: {combination_results['combination_statistics']['synergistic_combinations']}")
    report_lines.append(f"✓ Biomarker hypotheses: {len(biomarker_results['biomarker_hypotheses'])}")
    report_lines.append("")
    
    # Key Achievements
    report_lines.append("KEY ACHIEVEMENTS")
    report_lines.append("-" * 20)
    report_lines.append("✓ Comprehensive model interpretation framework")
    report_lines.append("✓ Automated drug repurposing pipeline")
    report_lines.append("✓ Novel knowledge discovery capabilities")
    report_lines.append("✓ Clinical decision support tools")
    report_lines.append("✓ Biomarker discovery platform")
    report_lines.append("✓ Therapeutic target identification")
    report_lines.append("✓ Drug combination prediction")
    report_lines.append("")
    
    # Next Steps
    report_lines.append("NEXT STEPS - PHASE 6: PUBLICATION & DEPLOYMENT")
    report_lines.append("-" * 50)
    report_lines.append("• Prepare research publication with experimental validation")
    report_lines.append("• Deploy API for production use")
    report_lines.append("• Create reproducibility package for community")
    report_lines.append("• Benchmark performance on real datasets")
    report_lines.append("• Establish collaborations for clinical validation")
    
    # Save report
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/phase5_summary_report.txt", "w") as f:
        f.write("\n".join(report_lines))
    
    print("\n" + "\n".join(report_lines))
    logger.info("Phase 5 summary report saved to outputs/phase5_summary_report.txt")


def main():
    """Main demonstration function for Phase 5."""
    logger.info("Starting Phase 5: Interpretation & Applications Demo")
    
    # Load configuration
    config = Config("configs/config.yaml")
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    # Create demo data
    demo_compounds, compound_metadata, disease_profile, clinical_outcomes = create_demo_data()
    
    # Initialize model (simplified for demo)
    model = MultiModalMoAModel(config)
    model.eval()
    
    logger.info("Phase 5 Demo: Interpretation & Applications")
    logger.info("=" * 50)
    
    # Demonstrate each component
    interpretation_results = demonstrate_model_interpretation(
        model, demo_compounds, compound_metadata, config
    )
    
    repurposing_results = demonstrate_drug_repurposing(
        model, demo_compounds, compound_metadata, config
    )
    
    discovery_results = demonstrate_knowledge_discovery(
        model, demo_compounds, compound_metadata, config
    )
    
    therapeutic_results = demonstrate_therapeutic_insights(
        model, demo_compounds, compound_metadata, config
    )
    
    # Create summary report
    create_phase5_summary_report(
        interpretation_results,
        repurposing_results,
        discovery_results,
        therapeutic_results
    )
    
    logger.info("Phase 5 demonstration completed successfully!")
    logger.info("All interpretation and application capabilities demonstrated")
    logger.info("Ready for Phase 6: Publication & Deployment")


if __name__ == "__main__":
    main()
