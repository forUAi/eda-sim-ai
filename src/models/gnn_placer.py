"""
Heterogeneous Graph Neural Network for chip placement.
Handles multiple node types (standard cells, IO ports) and edge types (nets).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, global_mean_pool
from torch_geometric.data import HeteroData
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class CellEncoder(nn.Module):
    """Encodes standard cell features into embeddings."""
    
    def __init__(self, 
                 num_cell_types: int,
                 cell_feat_dim: int,
                 hidden_dim: int):
        super().__init__()
        self.cell_type_embed = nn.Embedding(num_cell_types, hidden_dim // 2)
        self.cell_feat_proj = nn.Linear(cell_feat_dim, hidden_dim // 2)
        self.merge = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, cell_types: torch.Tensor, cell_features: torch.Tensor) -> torch.Tensor:
        type_emb = self.cell_type_embed(cell_types)
        feat_emb = self.cell_feat_proj(cell_features)
        merged = torch.cat([type_emb, feat_emb], dim=-1)
        return self.merge(merged)


class HeteroGNNLayer(nn.Module):
    """Single layer of heterogeneous GNN with attention mechanism."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        
        # Define convolutions for each edge type
        self.convs = HeteroConv({
            ('cell', 'net', 'cell'): GATConv(
                hidden_dim, hidden_dim // num_heads, 
                heads=num_heads, dropout=0.1, add_self_loops=False
            ),
            ('cell', 'net', 'io_port'): GATConv(
                hidden_dim, hidden_dim // num_heads,
                heads=num_heads, dropout=0.1, add_self_loops=False
            ),
            ('io_port', 'net', 'cell'): GATConv(
                hidden_dim, hidden_dim // num_heads,
                heads=num_heads, dropout=0.1, add_self_loops=False
            ),
        }, aggr='sum')
        
        # Node-wise transformations
        self.cell_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.io_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x_dict: Dict[str, torch.Tensor], 
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
                edge_attr_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        
        # Apply heterogeneous convolutions
        out_dict = self.convs(x_dict, edge_index_dict, edge_attr_dict)
        
        # Apply node-wise transformations with residual connections
        if 'cell' in out_dict:
            out_dict['cell'] = out_dict['cell'] + self.cell_transform(out_dict['cell'])
        if 'io_port' in out_dict:
            out_dict['io_port'] = out_dict['io_port'] + self.io_transform(out_dict['io_port'])
            
        return out_dict


class PlacementDecoder(nn.Module):
    """Decodes node embeddings into placement coordinates."""
    
    def __init__(self, hidden_dim: int, grid_size: int = 1000):
        super().__init__()
        self.grid_size = grid_size
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2)  # (x, y) coordinates
        )
        
        # Learnable grid embedding for spatial awareness
        self.grid_embed = nn.Parameter(
            torch.randn(1, 2, grid_size, grid_size) * 0.01
        )
        
    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_embeddings: [N, hidden_dim]
        
        Returns:
            placements: [N, 2] coordinates in range [0, grid_size]
        """
        raw_coords = self.decoder(node_embeddings)
        # Sigmoid to ensure coordinates are in [0, 1], then scale
        coords = torch.sigmoid(raw_coords) * self.grid_size
        return coords


class PPAHead(nn.Module):
    """Estimates PPA metrics from placed graph."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        # Global graph pooling
        self.graph_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Metric-specific heads
        self.wirelength_head = nn.Linear(hidden_dim // 2, 1)
        self.timing_head = nn.Linear(hidden_dim // 2, 2)  # WNS, TNS
        self.power_head = nn.Linear(hidden_dim // 2, 1)
        self.congestion_head = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, node_embeddings: torch.Tensor, batch: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Global pooling
        if batch is not None:
            pooled = global_mean_pool(node_embeddings, batch)
        else:
            pooled = node_embeddings.mean(dim=0, keepdim=True)
            
        graph_repr = self.graph_pool(pooled)
        
        metrics = {
            'hpwl': self.wirelength_head(graph_repr),
            'timing': self.timing_head(graph_repr),  # [WNS, TNS]
            'power': self.power_head(graph_repr),
            'congestion': self.congestion_head(graph_repr)
        }
        
        return metrics


class PlacementGNN(nn.Module):
    """Complete GNN model for chip placement."""
    
    def __init__(self,
                 num_cell_types: int,
                 cell_feat_dim: int,
                 io_feat_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 8,
                 num_heads: int = 4,
                 grid_size: int = 1000,
                 estimate_ppa: bool = True):
        super().__init__()
        
        self.num_layers = num_layers
        self.estimate_ppa = estimate_ppa
        
        # Feature encoders
        self.cell_encoder = CellEncoder(num_cell_types, cell_feat_dim, hidden_dim)
        self.io_encoder = nn.Sequential(
            nn.Linear(io_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            HeteroGNNLayer(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        
        # Decoders
        self.placement_decoder = PlacementDecoder(hidden_dim, grid_size)
        if estimate_ppa:
            self.ppa_head = PPAHead(hidden_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.1)
            
    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """
        Args:
            data: HeteroData with node features and edge indices
            
        Returns:
            Dictionary with 'placement' and optionally 'ppa_metrics'
        """
        # Encode node features
        x_dict = {
            'cell': self.cell_encoder(
                data['cell'].cell_type,
                data['cell'].x
            ),
            'io_port': self.io_encoder(data['io_port'].x)
        }
        
        # Apply GNN layers
        for layer in self.gnn_layers:
            x_dict = layer(x_dict, data.edge_index_dict)
            
        # Decode placements
        cell_placements = self.placement_decoder(x_dict['cell'])
        
        outputs = {'placement': cell_placements}
        
        # Estimate PPA if requested
        if self.estimate_ppa:
            # Combine all node embeddings for PPA estimation
            all_embeddings = torch.cat([x_dict['cell'], x_dict['io_port']], dim=0)
            ppa_metrics = self.ppa_head(all_embeddings)
            outputs['ppa_metrics'] = ppa_metrics
            
        return outputs
    
    def get_loss(self, outputs: Dict[str, torch.Tensor], 
                 targets: Dict[str, torch.Tensor],
                 loss_weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """Calculate weighted loss for training."""
        
        if loss_weights is None:
            loss_weights = {
                'placement': 1.0,
                'hpwl': 0.1,
                'timing': 0.2,
                'power': 0.1,
                'congestion': 0.1
            }
            
        total_loss = 0.0
        loss_dict = {}
        
        # Placement loss (MSE)
        if 'placement' in outputs and 'placement' in targets:
            placement_loss = F.mse_loss(outputs['placement'], targets['placement'])
            total_loss += loss_weights['placement'] * placement_loss
            loss_dict['placement'] = placement_loss.item()
            
        # PPA losses
        if 'ppa_metrics' in outputs and self.estimate_ppa:
            for metric in ['hpwl', 'power', 'congestion']:
                if metric in targets:
                    metric_loss = F.mse_loss(outputs['ppa_metrics'][metric], targets[metric])
                    total_loss += loss_weights.get(metric, 0.1) * metric_loss
                    loss_dict[metric] = metric_loss.item()
                    
            # Timing loss (WNS, TNS)
            if 'timing' in targets:
                timing_loss = F.mse_loss(outputs['ppa_metrics']['timing'], targets['timing'])
                total_loss += loss_weights.get('timing', 0.2) * timing_loss
                loss_dict['timing'] = timing_loss.item()
                
        return total_loss, loss_dict


class PlacementGNNWithConstraints(PlacementGNN):
    """Extended GNN model with placement constraints handling."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Additional modules for constraint handling
        self.constraint_encoder = nn.Sequential(
            nn.Linear(4, 64),  # [x_min, y_min, x_max, y_max]
            nn.ReLU(),
            nn.Linear(64, self.placement_decoder.decoder[0].in_features)
        )
        
    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        outputs = super().forward(data)
        
        # Apply placement constraints if present
        if hasattr(data['cell'], 'constraints'):
            constraints = data['cell'].constraints
            constraint_features = self.constraint_encoder(constraints)
            
            # Modify placements based on constraints
            placements = outputs['placement']
            x_min, y_min = constraints[:, 0], constraints[:, 1]
            x_max, y_max = constraints[:, 2], constraints[:, 3]
            
            # Clip to constraint bounds
            placements[:, 0] = torch.clamp(placements[:, 0], x_min, x_max)
            placements[:, 1] = torch.clamp(placements[:, 1], y_min, y_max)
            
            outputs['placement'] = placements
            
        return outputs