"""
Mechanism Tokens (MechTokens) - Novel ontology-aware embeddings.

This module implements MechTokens, a novel approach to encode drug-target-pathway-MoA
relationships using graph embeddings and hierarchical encoding.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from node2vec import Node2Vec
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn

from moa.utils.config import Config
from moa.utils.logger import get_logger

logger = get_logger(__name__)


class BiologicalOntologyGraph:
    """Build and manage biological ontology graph from multiple sources."""
    
    def __init__(self, config: Config):
        """
        Initialize ontology graph builder.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.ontology_sources = config.get("features.mechanism_tokens.ontology_sources", [])
        self.graph = nx.MultiDiGraph()
        self.node_types = {}
        self.hierarchical_levels = {}
        
    def add_chembl_data(self, mechanisms_df: pd.DataFrame, targets_df: pd.DataFrame) -> None:
        """
        Add ChEMBL mechanism and target data to ontology graph.
        
        Args:
            mechanisms_df: ChEMBL mechanisms DataFrame
            targets_df: ChEMBL targets DataFrame
        """
        logger.info("Adding ChEMBL data to ontology graph...")
        
        # Add drug-target-MoA relationships
        for _, row in mechanisms_df.iterrows():
            drug_id = row.get('molecule_chembl_id')
            target_id = row.get('target_chembl_id')
            moa = row.get('mechanism_of_action', '').strip().lower()
            
            if not all([drug_id, target_id, moa]):
                continue
            
            # Add nodes
            self.graph.add_node(drug_id, type='drug', level=0)
            self.graph.add_node(target_id, type='target', level=1)
            self.graph.add_node(f"moa_{moa}", type='moa', level=3)
            
            # Add edges
            self.graph.add_edge(drug_id, target_id, relation='targets')
            self.graph.add_edge(target_id, f"moa_{moa}", relation='mechanism')
            
            # Store node types
            self.node_types[drug_id] = 'drug'
            self.node_types[target_id] = 'target'
            self.node_types[f"moa_{moa}"] = 'moa'
            
            # Store hierarchical levels
            self.hierarchical_levels[drug_id] = 0
            self.hierarchical_levels[target_id] = 1
            self.hierarchical_levels[f"moa_{moa}"] = 3
        
        # Add target information
        for _, row in targets_df.iterrows():
            target_id = row.get('target_chembl_id')
            target_name = row.get('pref_name', '')
            target_type = row.get('target_type', '')
            
            if target_id in self.graph:
                self.graph.nodes[target_id]['name'] = target_name
                self.graph.nodes[target_id]['target_type'] = target_type
    
    def add_reactome_pathways(self, pathways_df: pd.DataFrame, protein_pathways_df: pd.DataFrame) -> None:
        """
        Add Reactome pathway data to ontology graph.
        
        Args:
            pathways_df: Reactome pathways DataFrame
            protein_pathways_df: Protein-pathway mappings DataFrame
        """
        logger.info("Adding Reactome pathways to ontology graph...")
        
        # Add pathway nodes
        for _, row in pathways_df.iterrows():
            pathway_id = row.get('stId')
            pathway_name = row.get('displayName', '')
            
            if pathway_id:
                self.graph.add_node(pathway_id, type='pathway', level=2, name=pathway_name)
                self.node_types[pathway_id] = 'pathway'
                self.hierarchical_levels[pathway_id] = 2
        
        # Add protein-pathway relationships
        for _, row in protein_pathways_df.iterrows():
            uniprot_id = row.get('uniprot_id')
            pathway_id = row.get('pathway_id')
            
            if uniprot_id and pathway_id:
                # Map UniProt to ChEMBL targets (simplified mapping)
                target_nodes = [n for n in self.graph.nodes() if self.node_types.get(n) == 'target']
                
                for target_id in target_nodes:
                    # Add pathway connection (simplified - in practice would need proper mapping)
                    if pathway_id in self.graph:
                        self.graph.add_edge(target_id, pathway_id, relation='participates_in')
    
    def add_hierarchical_relationships(self) -> None:
        """Add hierarchical relationships between different levels."""
        logger.info("Adding hierarchical relationships...")
        
        # Group nodes by type and level
        nodes_by_level = {}
        for node, level in self.hierarchical_levels.items():
            if level not in nodes_by_level:
                nodes_by_level[level] = []
            nodes_by_level[level].append(node)
        
        # Add pathway-MoA relationships based on shared targets
        if 2 in nodes_by_level and 3 in nodes_by_level:  # pathways and MoAs
            pathway_nodes = nodes_by_level[2]
            moa_nodes = nodes_by_level[3]
            
            for pathway in pathway_nodes:
                for moa in moa_nodes:
                    # Check if they share targets
                    pathway_targets = set(self.graph.predecessors(pathway))
                    moa_targets = set()
                    
                    # Get targets that lead to this MoA
                    for pred in self.graph.predecessors(moa):
                        if self.node_types.get(pred) == 'target':
                            moa_targets.add(pred)
                    
                    # If they share targets, add pathway-MoA edge
                    shared_targets = pathway_targets & moa_targets
                    if shared_targets:
                        self.graph.add_edge(pathway, moa, relation='leads_to', shared_targets=len(shared_targets))
    
    def build_ontology_graph(self, data_sources: Dict[str, pd.DataFrame]) -> nx.MultiDiGraph:
        """
        Build complete ontology graph from all sources.
        
        Args:
            data_sources: Dictionary of DataFrames from different sources
            
        Returns:
            NetworkX MultiDiGraph representing the biological ontology
        """
        logger.info("Building biological ontology graph...")
        
        # Add data from different sources
        if "chembl" in self.ontology_sources:
            if "mechanisms" in data_sources and "targets" in data_sources:
                self.add_chembl_data(data_sources["mechanisms"], data_sources["targets"])
        
        if "reactome" in self.ontology_sources:
            if "pathways" in data_sources and "protein_pathways" in data_sources:
                self.add_reactome_pathways(data_sources["pathways"], data_sources["protein_pathways"])
        
        # Add hierarchical relationships
        self.add_hierarchical_relationships()
        
        logger.info(f"Built ontology graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
        return self.graph


class MechTokenEmbedder:
    """Generate embeddings for mechanism tokens using node2vec and hierarchical encoding."""
    
    def __init__(self, config: Config):
        """
        Initialize MechToken embedder.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.embedding_dim = config.get("features.mechanism_tokens.embedding_dim", 256)
        self.node2vec_params = config.get("features.mechanism_tokens.node2vec_params", {})
        self.hierarchical_encoding = config.get("features.mechanism_tokens.hierarchical_encoding", True)
        
        self.embeddings = {}
        self.node_encoder = LabelEncoder()
        
    def train_node2vec_embeddings(self, graph: nx.MultiDiGraph) -> Dict[str, np.ndarray]:
        """
        Train node2vec embeddings on the ontology graph.
        
        Args:
            graph: Biological ontology graph
            
        Returns:
            Dictionary mapping node IDs to embeddings
        """
        logger.info("Training node2vec embeddings...")
        
        # Convert to simple graph for node2vec
        simple_graph = nx.Graph()
        for u, v, data in graph.edges(data=True):
            simple_graph.add_edge(u, v)
        
        # Configure node2vec parameters
        dimensions = self.node2vec_params.get("dimensions", 128)
        walk_length = self.node2vec_params.get("walk_length", 80)
        num_walks = self.node2vec_params.get("num_walks", 10)
        workers = self.node2vec_params.get("workers", 4)
        p = self.node2vec_params.get("p", 1.0)
        q = self.node2vec_params.get("q", 0.5)
        
        # Train node2vec
        node2vec = Node2Vec(
            simple_graph,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            workers=workers,
            p=p,
            q=q
        )
        
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        
        # Extract embeddings
        embeddings = {}
        for node in graph.nodes():
            if str(node) in model.wv:
                embeddings[node] = model.wv[str(node)]
            else:
                # Random embedding for missing nodes
                embeddings[node] = np.random.normal(0, 0.1, dimensions)
        
        logger.info(f"Generated node2vec embeddings for {len(embeddings)} nodes")
        return embeddings
    
    def add_hierarchical_encoding(
        self, 
        embeddings: Dict[str, np.ndarray], 
        node_types: Dict[str, str],
        hierarchical_levels: Dict[str, int]
    ) -> Dict[str, np.ndarray]:
        """
        Add hierarchical positional encoding to embeddings.
        
        Args:
            embeddings: Base node2vec embeddings
            node_types: Node type mappings
            hierarchical_levels: Hierarchical level mappings
            
        Returns:
            Enhanced embeddings with hierarchical encoding
        """
        if not self.hierarchical_encoding:
            return embeddings
        
        logger.info("Adding hierarchical positional encoding...")
        
        enhanced_embeddings = {}
        
        # Create type and level encodings
        unique_types = list(set(node_types.values()))
        unique_levels = list(set(hierarchical_levels.values()))
        
        type_encoder = LabelEncoder()
        type_encoder.fit(unique_types)
        
        for node, embedding in embeddings.items():
            enhanced_emb = embedding.copy()
            
            # Add type encoding
            if node in node_types:
                node_type = node_types[node]
                type_encoding = type_encoder.transform([node_type])[0]
                type_vector = np.zeros(len(unique_types))
                type_vector[type_encoding] = 1.0
                enhanced_emb = np.concatenate([enhanced_emb, type_vector])
            
            # Add level encoding
            if node in hierarchical_levels:
                level = hierarchical_levels[node]
                level_vector = np.zeros(max(unique_levels) + 1)
                level_vector[level] = 1.0
                enhanced_emb = np.concatenate([enhanced_emb, level_vector])
            
            enhanced_embeddings[node] = enhanced_emb
        
        return enhanced_embeddings
    
    def generate_mechanism_tokens(
        self, 
        graph: nx.MultiDiGraph, 
        node_types: Dict[str, str],
        hierarchical_levels: Dict[str, int]
    ) -> Dict[str, np.ndarray]:
        """
        Generate complete mechanism token embeddings.
        
        Args:
            graph: Biological ontology graph
            node_types: Node type mappings
            hierarchical_levels: Hierarchical level mappings
            
        Returns:
            Dictionary of mechanism token embeddings
        """
        logger.info("Generating mechanism tokens...")
        
        # Train base node2vec embeddings
        base_embeddings = self.train_node2vec_embeddings(graph)
        
        # Add hierarchical encoding
        mechanism_tokens = self.add_hierarchical_encoding(
            base_embeddings, node_types, hierarchical_levels
        )
        
        self.embeddings = mechanism_tokens
        
        logger.info(f"Generated mechanism tokens with dimension {list(mechanism_tokens.values())[0].shape[0]}")
        return mechanism_tokens


class MechTokenFeatureExtractor:
    """Extract mechanism token features for compounds."""
    
    def __init__(self, config: Config):
        """
        Initialize MechToken feature extractor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.ontology_builder = BiologicalOntologyGraph(config)
        self.embedder = MechTokenEmbedder(config)
        
        self.ontology_graph = None
        self.mechanism_tokens = None
        
    def build_mechanism_tokens(self, data_sources: Dict[str, pd.DataFrame]) -> None:
        """
        Build mechanism tokens from data sources.
        
        Args:
            data_sources: Dictionary of DataFrames from different sources
        """
        logger.info("Building mechanism tokens...")
        
        # Build ontology graph
        self.ontology_graph = self.ontology_builder.build_ontology_graph(data_sources)
        
        # Generate mechanism token embeddings
        self.mechanism_tokens = self.embedder.generate_mechanism_tokens(
            self.ontology_graph,
            self.ontology_builder.node_types,
            self.ontology_builder.hierarchical_levels
        )
    
    def extract_compound_tokens(self, compound_ids: List[str]) -> Dict[str, np.ndarray]:
        """
        Extract mechanism tokens for specific compounds.
        
        Args:
            compound_ids: List of compound ChEMBL IDs
            
        Returns:
            Dictionary mapping compound IDs to aggregated mechanism tokens
        """
        if self.mechanism_tokens is None:
            raise ValueError("Mechanism tokens not built. Call build_mechanism_tokens first.")
        
        logger.info(f"Extracting mechanism tokens for {len(compound_ids)} compounds...")
        
        compound_tokens = {}
        
        for compound_id in compound_ids:
            if compound_id in self.ontology_graph:
                # Get all connected nodes (targets, pathways, MoAs)
                connected_nodes = list(nx.descendants(self.ontology_graph, compound_id))
                connected_nodes.append(compound_id)  # Include the compound itself
                
                # Aggregate embeddings from connected nodes
                embeddings = []
                for node in connected_nodes:
                    if node in self.mechanism_tokens:
                        embeddings.append(self.mechanism_tokens[node])
                
                if embeddings:
                    # Use mean aggregation (could also use attention-based aggregation)
                    compound_tokens[compound_id] = np.mean(embeddings, axis=0)
                else:
                    # Random token for compounds without connections
                    token_dim = list(self.mechanism_tokens.values())[0].shape[0]
                    compound_tokens[compound_id] = np.random.normal(0, 0.1, token_dim)
            else:
                # Random token for unknown compounds
                token_dim = list(self.mechanism_tokens.values())[0].shape[0]
                compound_tokens[compound_id] = np.random.normal(0, 0.1, token_dim)
        
        return compound_tokens
    
    def get_moa_tokens(self, moa_names: List[str]) -> Dict[str, np.ndarray]:
        """
        Get mechanism tokens for specific MoAs.
        
        Args:
            moa_names: List of MoA names
            
        Returns:
            Dictionary mapping MoA names to tokens
        """
        if self.mechanism_tokens is None:
            raise ValueError("Mechanism tokens not built. Call build_mechanism_tokens first.")
        
        moa_tokens = {}
        
        for moa_name in moa_names:
            moa_key = f"moa_{moa_name.strip().lower()}"
            if moa_key in self.mechanism_tokens:
                moa_tokens[moa_name] = self.mechanism_tokens[moa_key]
            else:
                # Random token for unknown MoAs
                token_dim = list(self.mechanism_tokens.values())[0].shape[0]
                moa_tokens[moa_name] = np.random.normal(0, 0.1, token_dim)
        
        return moa_tokens
    
    def save_mechanism_tokens(self, save_path: Path) -> None:
        """Save mechanism tokens to file."""
        if self.mechanism_tokens is None:
            raise ValueError("No mechanism tokens to save")
        
        np.savez(save_path, **self.mechanism_tokens)
        logger.info(f"Saved mechanism tokens to {save_path}")
    
    def load_mechanism_tokens(self, load_path: Path) -> None:
        """Load mechanism tokens from file."""
        data = np.load(load_path)
        self.mechanism_tokens = {key: data[key] for key in data.files}
        logger.info(f"Loaded mechanism tokens from {load_path}")
