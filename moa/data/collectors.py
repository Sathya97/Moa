"""Data collection utilities for various databases."""

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import requests
from tqdm import tqdm

from moa.utils.config import Config
from moa.utils.logger import get_logger

logger = get_logger(__name__)


class DataCollector(ABC):
    """Abstract base class for data collectors."""
    
    def __init__(self, config: Config, cache_dir: Optional[Path] = None):
        """
        Initialize data collector.
        
        Args:
            config: Configuration object
            cache_dir: Directory for caching downloaded data
        """
        self.config = config
        self.cache_dir = cache_dir or Path("cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    @abstractmethod
    def collect(self, **kwargs) -> pd.DataFrame:
        """Collect data from the source."""
        pass
    
    def _make_request(self, url: str, params: Optional[Dict] = None, 
                     timeout: int = 30, retries: int = 3) -> requests.Response:
        """
        Make HTTP request with retries.
        
        Args:
            url: Request URL
            params: Query parameters
            timeout: Request timeout
            retries: Number of retries
            
        Returns:
            Response object
        """
        for attempt in range(retries):
            try:
                response = requests.get(url, params=params, timeout=timeout)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                if attempt == retries - 1:
                    raise e
                logger.warning(f"Request failed (attempt {attempt + 1}/{retries}): {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
        
        raise requests.RequestException("Max retries exceeded")


class ChEMBLCollector(DataCollector):
    """Collector for ChEMBL database."""
    
    def __init__(self, config: Config, cache_dir: Optional[Path] = None):
        super().__init__(config, cache_dir)
        self.base_url = config.get("chembl.base_url", "https://www.ebi.ac.uk/chembl/api/data")
        self.version = config.get("chembl.version", "33")
        
    def collect_mechanisms(self) -> pd.DataFrame:
        """
        Collect mechanism of action data from ChEMBL.
        
        Returns:
            DataFrame with mechanism data
        """
        logger.info("Collecting ChEMBL mechanism data...")
        
        cache_file = self.cache_dir / f"chembl_mechanisms_v{self.version}.pkl"
        if cache_file.exists():
            logger.info(f"Loading cached mechanisms from {cache_file}")
            return pd.read_pickle(cache_file)
        
        mechanisms = []
        offset = 0
        limit = 1000
        
        while True:
            url = f"{self.base_url}/mechanism"
            params = {
                "limit": limit,
                "offset": offset,
                "format": "json"
            }
            
            response = self._make_request(url, params)
            data = response.json()
            
            if not data.get("mechanisms"):
                break
                
            mechanisms.extend(data["mechanisms"])
            offset += limit
            
            logger.info(f"Collected {len(mechanisms)} mechanisms...")
            
            if len(data["mechanisms"]) < limit:
                break
        
        df = pd.DataFrame(mechanisms)
        df.to_pickle(cache_file)
        logger.info(f"Collected {len(df)} mechanisms from ChEMBL")
        
        return df
    
    def collect_activities(self, target_chembl_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Collect activity data from ChEMBL.
        
        Args:
            target_chembl_ids: Optional list of target IDs to filter by
            
        Returns:
            DataFrame with activity data
        """
        logger.info("Collecting ChEMBL activity data...")
        
        cache_file = self.cache_dir / f"chembl_activities_v{self.version}.pkl"
        if cache_file.exists() and target_chembl_ids is None:
            logger.info(f"Loading cached activities from {cache_file}")
            return pd.read_pickle(cache_file)
        
        activities = []
        offset = 0
        limit = 1000
        
        # Activity filters from config
        activity_types = self.config.get("chembl.activities.activity_types", [])
        pchembl_min = self.config.get("chembl.activities.pchembl_value_min", 4.0)
        
        while True:
            url = f"{self.base_url}/activity"
            params = {
                "limit": limit,
                "offset": offset,
                "format": "json",
                "pchembl_value__gte": pchembl_min,
                "standard_relation__in": "=,<,<="
            }
            
            if activity_types:
                params["standard_type__in"] = ",".join(activity_types)
            
            if target_chembl_ids:
                params["target_chembl_id__in"] = ",".join(target_chembl_ids)
            
            response = self._make_request(url, params)
            data = response.json()
            
            if not data.get("activities"):
                break
                
            activities.extend(data["activities"])
            offset += limit
            
            logger.info(f"Collected {len(activities)} activities...")
            
            if len(data["activities"]) < limit:
                break
        
        df = pd.DataFrame(activities)
        if target_chembl_ids is None:
            df.to_pickle(cache_file)
        
        logger.info(f"Collected {len(df)} activities from ChEMBL")
        return df
    
    def collect_targets(self) -> pd.DataFrame:
        """
        Collect target data from ChEMBL.
        
        Returns:
            DataFrame with target data
        """
        logger.info("Collecting ChEMBL target data...")
        
        cache_file = self.cache_dir / f"chembl_targets_v{self.version}.pkl"
        if cache_file.exists():
            logger.info(f"Loading cached targets from {cache_file}")
            return pd.read_pickle(cache_file)
        
        targets = []
        offset = 0
        limit = 1000
        
        target_types = self.config.get("chembl.targets.target_types", ["PROTEIN"])
        organism = self.config.get("chembl.targets.organism", "Homo sapiens")
        
        while True:
            url = f"{self.base_url}/target"
            params = {
                "limit": limit,
                "offset": offset,
                "format": "json",
                "target_type__in": ",".join(target_types),
                "organism__icontains": organism
            }
            
            response = self._make_request(url, params)
            data = response.json()
            
            if not data.get("targets"):
                break
                
            targets.extend(data["targets"])
            offset += limit
            
            logger.info(f"Collected {len(targets)} targets...")
            
            if len(data["targets"]) < limit:
                break
        
        df = pd.DataFrame(targets)
        df.to_pickle(cache_file)
        logger.info(f"Collected {len(df)} targets from ChEMBL")
        
        return df
    
    def collect_compounds(self, molecule_chembl_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Collect compound data from ChEMBL.
        
        Args:
            molecule_chembl_ids: Optional list of molecule IDs to filter by
            
        Returns:
            DataFrame with compound data
        """
        logger.info("Collecting ChEMBL compound data...")
        
        cache_file = self.cache_dir / f"chembl_compounds_v{self.version}.pkl"
        if cache_file.exists() and molecule_chembl_ids is None:
            logger.info(f"Loading cached compounds from {cache_file}")
            return pd.read_pickle(cache_file)
        
        compounds = []
        
        if molecule_chembl_ids:
            # Batch requests for specific molecules
            batch_size = 50
            for i in tqdm(range(0, len(molecule_chembl_ids), batch_size)):
                batch_ids = molecule_chembl_ids[i:i + batch_size]
                url = f"{self.base_url}/molecule"
                params = {
                    "molecule_chembl_id__in": ",".join(batch_ids),
                    "format": "json"
                }
                
                response = self._make_request(url, params)
                data = response.json()
                
                if data.get("molecules"):
                    compounds.extend(data["molecules"])
        else:
            # Collect all compounds (this will be large!)
            offset = 0
            limit = 1000
            
            max_heavy_atoms = self.config.get("chembl.compounds.max_heavy_atoms", 100)
            min_heavy_atoms = self.config.get("chembl.compounds.min_heavy_atoms", 5)
            
            while True:
                url = f"{self.base_url}/molecule"
                params = {
                    "limit": limit,
                    "offset": offset,
                    "format": "json",
                    "molecule_properties__heavy_atoms__lte": max_heavy_atoms,
                    "molecule_properties__heavy_atoms__gte": min_heavy_atoms
                }
                
                response = self._make_request(url, params)
                data = response.json()
                
                if not data.get("molecules"):
                    break
                    
                compounds.extend(data["molecules"])
                offset += limit
                
                logger.info(f"Collected {len(compounds)} compounds...")
                
                if len(data["molecules"]) < limit:
                    break
        
        df = pd.DataFrame(compounds)
        if molecule_chembl_ids is None:
            df.to_pickle(cache_file)
        
        logger.info(f"Collected {len(df)} compounds from ChEMBL")
        return df
    
    def collect(self, data_types: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Collect multiple data types from ChEMBL.
        
        Args:
            data_types: List of data types to collect
            
        Returns:
            Dictionary of DataFrames
        """
        if data_types is None:
            data_types = ["mechanisms", "activities", "targets", "compounds"]
        
        results = {}
        
        for data_type in data_types:
            if data_type == "mechanisms":
                results["mechanisms"] = self.collect_mechanisms()
            elif data_type == "activities":
                results["activities"] = self.collect_activities()
            elif data_type == "targets":
                results["targets"] = self.collect_targets()
            elif data_type == "compounds":
                results["compounds"] = self.collect_compounds()
            else:
                logger.warning(f"Unknown data type: {data_type}")
        
        return results


class LINCSCollector(DataCollector):
    """Collector for LINCS L1000 data."""

    def __init__(self, config: Config, cache_dir: Optional[Path] = None):
        super().__init__(config, cache_dir)
        self.base_url = config.get("lincs.base_url", "https://clue.io/api")

    def collect_signatures(self, compound_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Collect gene expression signatures from LINCS.

        Args:
            compound_names: Optional list of compound names to filter by

        Returns:
            DataFrame with signature data
        """
        logger.info("Collecting LINCS signature data...")

        cache_file = self.cache_dir / "lincs_signatures.pkl"
        if cache_file.exists() and compound_names is None:
            logger.info(f"Loading cached signatures from {cache_file}")
            return pd.read_pickle(cache_file)

        # This is a simplified implementation
        # In practice, you would use the LINCS API or download data files
        logger.warning("LINCS collector is a placeholder - implement with actual API")

        # Placeholder data structure
        signatures = pd.DataFrame({
            'sig_id': [],
            'pert_iname': [],
            'cell_id': [],
            'pert_time': [],
            'pert_dose': [],
            'gene_expression': []
        })

        return signatures

    def collect(self, **kwargs) -> pd.DataFrame:
        """Collect LINCS data."""
        return self.collect_signatures(**kwargs)


class ReactomeCollector(DataCollector):
    """Collector for Reactome pathway data."""

    def __init__(self, config: Config, cache_dir: Optional[Path] = None):
        super().__init__(config, cache_dir)
        self.base_url = config.get("reactome.base_url", "https://reactome.org/ContentService")
        self.species = config.get("reactome.species", "Homo sapiens")

    def collect_pathways(self) -> pd.DataFrame:
        """
        Collect pathway data from Reactome.

        Returns:
            DataFrame with pathway data
        """
        logger.info("Collecting Reactome pathway data...")

        cache_file = self.cache_dir / "reactome_pathways.pkl"
        if cache_file.exists():
            logger.info(f"Loading cached pathways from {cache_file}")
            return pd.read_pickle(cache_file)

        # Get top-level pathways
        url = f"{self.base_url}/data/pathways/top/{self.species}"
        response = self._make_request(url)
        pathways = response.json()

        # Expand to get all pathway levels
        all_pathways = []
        for pathway in tqdm(pathways, desc="Collecting pathways"):
            all_pathways.append(pathway)

            # Get child pathways
            children_url = f"{self.base_url}/data/pathway/{pathway['stId']}/containedEvents"
            try:
                children_response = self._make_request(children_url)
                children = children_response.json()
                all_pathways.extend(children)
            except requests.RequestException:
                continue

        df = pd.DataFrame(all_pathways)
        df.to_pickle(cache_file)
        logger.info(f"Collected {len(df)} pathways from Reactome")

        return df

    def collect_protein_pathways(self, uniprot_ids: List[str]) -> pd.DataFrame:
        """
        Collect protein-pathway mappings from Reactome.

        Args:
            uniprot_ids: List of UniProt IDs

        Returns:
            DataFrame with protein-pathway mappings
        """
        logger.info("Collecting protein-pathway mappings from Reactome...")

        mappings = []
        for uniprot_id in tqdm(uniprot_ids, desc="Mapping proteins to pathways"):
            url = f"{self.base_url}/data/pathways/low/entity/{uniprot_id}"
            try:
                response = self._make_request(url)
                pathways = response.json()

                for pathway in pathways:
                    mappings.append({
                        'uniprot_id': uniprot_id,
                        'pathway_id': pathway.get('stId'),
                        'pathway_name': pathway.get('displayName'),
                        'species': pathway.get('species', [{}])[0].get('displayName')
                    })
            except requests.RequestException:
                continue

        df = pd.DataFrame(mappings)
        logger.info(f"Collected {len(df)} protein-pathway mappings")

        return df

    def collect(self, **kwargs) -> Dict[str, pd.DataFrame]:
        """Collect Reactome data."""
        results = {
            'pathways': self.collect_pathways()
        }

        if 'uniprot_ids' in kwargs:
            results['protein_pathways'] = self.collect_protein_pathways(kwargs['uniprot_ids'])

        return results


class DataCollectorFactory:
    """Factory for creating data collectors."""

    @staticmethod
    def create_collector(source: str, config: Config, cache_dir: Optional[Path] = None) -> DataCollector:
        """
        Create a data collector for the specified source.

        Args:
            source: Data source name
            config: Configuration object
            cache_dir: Cache directory

        Returns:
            Data collector instance
        """
        collectors = {
            'chembl': ChEMBLCollector,
            'lincs': LINCSCollector,
            'reactome': ReactomeCollector,
        }

        if source not in collectors:
            raise ValueError(f"Unknown data source: {source}")

        return collectors[source](config, cache_dir)
