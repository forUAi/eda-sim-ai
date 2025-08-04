"""
Feature extraction for chip placement graphs.
Converts netlists into rich graph representations with node and edge features.
"""

import numpy as np
import torch
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import networkx as nx
from collections import defaultdict
import logging

from .netlist_parser import NetlistGraph, Cell, Net, IOPort

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    # Cell features
    use_cell_type: bool = True
    use_cell_area: bool = True
    use_pin_count: bool = True
    use_timing_criticality: bool = True
    use_power_features: bool = True
    
    # Net features  
    use_fanout: bool = True
    use_net_criticality: bool = True
    use_estimated_capacitance: bool = True
    
    # Graph features
    use_spectral_features: bool = True
    spectral_dim: int = 16
    
    # Normalization
    normalize_features: bool = True
    
    # Cell type encoding
    cell_type_vocab: Optional[Dict[str, int]] = None
    max_cell_types: int = 100


class FeatureExtractor:
    """Extracts features from netlist graphs for GNN processing."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.cell_type_encoder = {}
        self.feature_stats = {}
        
    def fit(self, graphs: List[NetlistGraph]) -> None:
        """Fit feature extractors on training data."""
        logger.info("Fitting feature extractors...")
        
        # Build cell type vocabulary
        if self.config.use_cell_type:
            self._build_cell_type_vocab(graphs)
            
        # Compute feature statistics for normalization
        if self.config.normalize_features:
            self._compute_feature_stats(graphs)
            
        logger.info(f"Fitted on {len(graphs)} graphs")
        
    def extract(self, graph: NetlistGraph, 
                placement: Optional[np.ndarray] = None,
                timing_info: Optional[Dict] = None) -> HeteroData:
        """Extract features from a single netlist graph."""
        
        data = HeteroData()
        
        # Extract node features
        cell_features, cell_indices = self._extract_cell_features(graph, timing_info)
        io_features, io_indices = self._extract_io_features(graph)
        
        # Set node features
        data['cell'].x = torch.FloatTensor(cell_features)
        data['cell'].cell_type = torch.LongTensor(
            [self._encode_cell_type(cell.cell_type) for cell in graph.cells]
        )
        
        data['io_port'].x = torch.FloatTensor(io_features)
        
        # Extract edge indices and features
        edge_dict, edge_features = self._extract_edges(graph, cell_indices, io_indices)
        
        for edge_type, edge_index in edge_dict.items():
            data[edge_type].edge_index = edge_index
            if edge_type in edge_features:
                data[edge_type].edge_attr = edge_features[edge_type]
                
        # Add placement if provided
        if placement is not None:
            data['cell'].pos = torch.FloatTensor(placement)
            
        # Add global graph features
        if self.config.use_spectral_features:
            data.graph_features = self._extract_spectral_features(graph)
            
        # Add metadata
        data.num_cells = len(graph.cells)
        data.num_nets = len(graph.nets)
        data.design_name = graph.design_name
        
        return data
    
    def _build_cell_type_vocab(self, graphs: List[NetlistGraph]) -> None:
        """Build vocabulary for cell types."""
        cell_types = set()
        
        for graph in graphs:
            for cell in graph.cells:
                cell_types.add(cell.cell_type)
                
        # Sort for consistency
        sorted_types = sorted(list(cell_types))
        
        # Create encoding (reserve 0 for unknown)
        self.cell_type_encoder = {
            cell_type: idx + 1 
            for idx, cell_type in enumerate(sorted_types[:self.config.max_cell_types - 1])
        }
        self.cell_type_encoder['<UNK>'] = 0
        
        logger.info(f"Built cell type vocabulary with {len(self.cell_type_encoder)} types")
        
    def _compute_feature_stats(self, graphs: List[NetlistGraph]) -> None:
        """Compute feature statistics for normalization."""
        all_cell_features = []
        all_io_features = []
        
        for graph in graphs:
            cell_features, _ = self._extract_cell_features(graph, None)
            io_features, _ = self._extract_io_features(graph)
            
            all_cell_features.append(cell_features)
            all_io_features.append(io_features)
            
        # Concatenate all features
        all_cell_features = np.vstack(all_cell_features)
        all_io_features = np.vstack(all_io_features)
        
        # Compute statistics
        self.feature_stats['cell'] = {
            'mean': all_cell_features.mean(axis=0),
            'std': all_cell_features.std(axis=0) + 1e-6
        }
        
        self.feature_stats['io'] = {
            'mean': all_io_features.mean(axis=0),
            'std': all_io_features.std(axis=0) + 1e-6
        }
        
    def _extract_cell_features(self, graph: NetlistGraph, 
                              timing_info: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, int]]:
        """Extract features for standard cells."""
        features = []
        indices = {}
        
        for idx, cell in enumerate(graph.cells):
            cell_feat = []
            
            # Basic features
            if self.config.use_cell_area:
                cell_feat.append(cell.area)
                
            if self.config.use_pin_count:
                cell_feat.extend([cell.num_inputs, cell.num_outputs])
                
            # Timing features
            if self.config.use_timing_criticality and timing_info:
                criticality = timing_info.get('cell_criticality', {}).get(cell.name, 0.0)
                cell_feat.append(criticality)
            else:
                cell_feat.append(0.0)
                
            # Power features
            if self.config.use_power_features:
                cell_feat.extend([
                    cell.static_power,
                    cell.switching_power,
                    cell.internal_power
                ])
            else:
                cell_feat.extend([0.0, 0.0, 0.0])
                
            # Connectivity features
            fanin = len([n for n in graph.nets if cell.name in n.pins])
            fanout = len([n for n in graph.nets if cell.name in n.driver])
            cell_feat.extend([fanin, fanout])
            
            features.append(cell_feat)
            indices[cell.name] = idx
            
        features = np.array(features, dtype=np.float32)
        
        # Normalize if configured
        if self.config.normalize_features and 'cell' in self.feature_stats:
            features = (features - self.feature_stats['cell']['mean']) / self.feature_stats['cell']['std']
            
        return features, indices
    
    def _extract_io_features(self, graph: NetlistGraph) -> Tuple[np.ndarray, Dict[str, int]]:
        """Extract features for IO ports."""
        features = []
        indices = {}
        
        for idx, io_port in enumerate(graph.io_ports):
            io_feat = []
            
            # Direction (input: 1, output: 0)
            io_feat.append(1.0 if io_port.direction == 'input' else 0.0)
            
            # Connectivity
            connected_nets = len([n for n in graph.nets if io_port.name in n.pins])
            io_feat.append(connected_nets)
            
            # Fixed position if available
            if io_port.x is not None and io_port.y is not None:
                io_feat.extend([io_port.x, io_port.y])
            else:
                io_feat.extend([0.0, 0.0])
                
            features.append(io_feat)
            indices[io_port.name] = idx
            
        features = np.array(features, dtype=np.float32)
        
        # Normalize if configured
        if self.config.normalize_features and 'io' in self.feature_stats:
            features = (features - self.feature_stats['io']['mean']) / self.feature_stats['io']['std']
            
        return features, indices
    
    def _extract_edges(self, graph: NetlistGraph,
                      cell_indices: Dict[str, int],
                      io_indices: Dict[str, int]) -> Tuple[Dict, Dict]:
        """Extract edge connections and features."""
        edge_dict = defaultdict(list)
        edge_features = defaultdict(list)
        
        for net in graph.nets:
            # Get all connected components
            connected_cells = []
            connected_ios = []
            
            for pin in net.pins:
                if pin in cell_indices:
                    connected_cells.append(cell_indices[pin])
                elif pin in io_indices:
                    connected_ios.append(io_indices[pin])
                    
            # Create edges between all connected components
            # Cell-to-cell edges
            for i in range(len(connected_cells)):
                for j in range(i + 1, len(connected_cells)):
                    edge_dict[('cell', 'net', 'cell')].append([connected_cells[i], connected_cells[j]])
                    edge_dict[('cell', 'net', 'cell')].append([connected_cells[j], connected_cells[i]])
                    
                    # Add edge features
                    if self.config.use_fanout or self.config.use_net_criticality:
                        edge_feat = []
                        if self.config.use_fanout:
                            edge_feat.append(len(net.pins))
                        if self.config.use_net_criticality:
                            edge_feat.append(net.criticality if hasattr(net, 'criticality') else 0.0)
                        if self.config.use_estimated_capacitance:
                            edge_feat.append(net.capacitance if hasattr(net, 'capacitance') else 0.0)
                            
                        edge_features[('cell', 'net', 'cell')].append(edge_feat)
                        edge_features[('cell', 'net', 'cell')].append(edge_feat)
                        
            # Cell-to-IO edges
            for cell_idx in connected_cells:
                for io_idx in connected_ios:
                    edge_dict[('cell', 'net', 'io_port')].append([cell_idx, io_idx])
                    edge_dict[('io_port', 'net', 'cell')].append([io_idx, cell_idx])
                    
        # Convert to tensors
        for edge_type in edge_dict:
            if edge_dict[edge_type]:
                edge_dict[edge_type] = torch.LongTensor(edge_dict[edge_type]).t()
            else:
                edge_dict[edge_type] = torch.LongTensor(0, 2)
                
        for edge_type in edge_features:
            if edge_features[edge_type]:
                edge_features[edge_type] = torch.FloatTensor(edge_features[edge_type])
                
        return edge_dict, edge_features
    
    def _extract_spectral_features(self, graph: NetlistGraph) -> torch.Tensor:
        """Extract spectral features from the netlist graph."""
        # Build NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        for cell in graph.cells:
            G.add_node(cell.name, node_type='cell')
        for io in graph.io_ports:
            G.add_node(io.name, node_type='io')
            
        # Add edges
        for net in graph.nets:
            pins = list(net.pins)
            for i in range(len(pins)):
                for j in range(i + 1, len(pins)):
                    G.add_edge(pins[i], pins[j])
                    
        # Compute Laplacian eigenvalues
        try:
            laplacian = nx.laplacian_matrix(G).astype(np.float32)
            eigenvalues = np.linalg.eigvalsh(laplacian.toarray())[:self.config.spectral_dim]
            
            # Pad if needed
            if len(eigenvalues) < self.config.spectral_dim:
                eigenvalues = np.pad(eigenvalues, 
                                   (0, self.config.spectral_dim - len(eigenvalues)),
                                   mode='constant')
        except:
            eigenvalues = np.zeros(self.config.spectral_dim)
            
        return torch.FloatTensor(eigenvalues)
    
    def _encode_cell_type(self, cell_type: str) -> int:
        """Encode cell type to integer."""
        return self.cell_type_encoder.get(cell_type, 0)  # 0 for unknown


class NetlistDataset(torch.utils.data.Dataset):
    """PyTorch dataset for netlist graphs."""
    
    def __init__(self, 
                 data_dir: str,
                 feature_extractor: FeatureExtractor,
                 transform=None,
                 pre_transform=None):
        super().__init__()
        
        self.data_dir = data_dir
        self.feature_extractor = feature_extractor
        self.transform = transform
        self.pre_transform = pre_transform
        
        # Load all graphs
        self.graphs = self._load_graphs()
        
        # Extract features for all graphs
        self.data_list = []
        for graph in self.graphs:
            data = self.feature_extractor.extract(graph)
            if self.pre_transform:
                data = self.pre_transform(data)
            self.data_list.append(data)
            
        logger.info(f"Loaded {len(self.data_list)} graphs")
        
    def _load_graphs(self) -> List[NetlistGraph]:
        """Load all netlist graphs from directory."""
        import os
        import json
        
        graphs = []
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json'):
                with open(os.path.join(self.data_dir, filename)) as f:
                    graph_data = json.load(f)
                    graph = NetlistGraph.from_dict(graph_data)
                    graphs.append(graph)
                    
        return graphs
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        
        if self.transform:
            data = self.transform(data)
            
        return data


class PlacementAugmentation:
    """Data augmentation for placement tasks."""
    
    def __init__(self, 
                 flip_prob: float = 0.5,
                 rotate_prob: float = 0.5,
                 noise_std: float = 0.01):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.noise_std = noise_std
        
    def __call__(self, data: HeteroData) -> HeteroData:
        """Apply augmentations to placement data."""
        if hasattr(data['cell'], 'pos'):
            pos = data['cell'].pos.clone()
            
            # Random flip
            if np.random.random() < self.flip_prob:
                if np.random.random() < 0.5:
                    pos[:, 0] = 1.0 - pos[:, 0]  # Flip X
                else:
                    pos[:, 1] = 1.0 - pos[:, 1]  # Flip Y
                    
            # Random rotation (90 degree increments)
            if np.random.random() < self.rotate_prob:
                angle = np.random.choice([90, 180, 270])
                if angle == 90:
                    pos = torch.stack([pos[:, 1], 1.0 - pos[:, 0]], dim=1)
                elif angle == 180:
                    pos = 1.0 - pos
                elif angle == 270:
                    pos = torch.stack([1.0 - pos[:, 1], pos[:, 0]], dim=1)
                    
            # Add noise
            if self.noise_std > 0:
                noise = torch.randn_like(pos) * self.noise_std
                pos = torch.clamp(pos + noise, 0, 1)
                
            data['cell'].pos = pos
            
        return data


def create_dataloaders(config: Dict, 
                      batch_size: int = 32,
                      num_workers: int = 4) -> Tuple[torch.utils.data.DataLoader, ...]:
    """Create train/val/test dataloaders."""
    from torch_geometric.loader import DataLoader
    
    # Create feature extractor
    feat_config = FeatureConfig(**config['features'])
    feature_extractor = FeatureExtractor(feat_config)
    
    # Load datasets
    train_dataset = NetlistDataset(
        config['train_dir'],
        feature_extractor,
        transform=PlacementAugmentation() if config.get('augment', True) else None
    )
    
    # Fit feature extractor on training data
    feature_extractor.fit([g for g in train_dataset.graphs])
    
    val_dataset = NetlistDataset(config['val_dir'], feature_extractor)
    test_dataset = NetlistDataset(config['test_dir'], feature_extractor)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    config = {
        'features': {
            'use_cell_type': True,
            'use_cell_area': True,
            'use_pin_count': True,
            'use_timing_criticality': True,
            'use_power_features': True,
            'use_fanout': True,
            'use_net_criticality': True,
            'use_spectral_features': True,
            'normalize_features': True
        },
        'train_dir': 'data/train',
        'val_dir': 'data/val',
        'test_dir': 'data/test',
        'augment': True
    }
    
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # Test iteration
    for batch in train_loader:
        print(f"Batch size: {batch.num_graphs}")
        print(f"Cell features: {batch['cell'].x.shape}")
        print(f"IO features: {batch['io_port'].x.shape}")
        break