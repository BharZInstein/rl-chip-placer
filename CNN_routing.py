import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import numpy as np

class ChipPlacementGNN(nn.Module):
    """
    Graph Neural Network for chip placement that processes netlists
    and outputs embeddings for placement decisions.
    
    The network takes a graph where:
    - Nodes = electronic components (gates, macros, I/O)
    - Edges = nets (wires connecting components)
    """
    
    def __init__(self, 
                 node_feature_dim=8,      # Input node features (size, type, position, etc.)
                 edge_feature_dim=4,      # Input edge features (net weight, timing, etc.)
                 hidden_dim=128,          # Hidden layer size
                 num_gnn_layers=4,        # Number of graph convolution layers
                 output_dim=64):          # Output embedding size
        
        super(ChipPlacementGNN, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.output_dim = output_dim
        
        # Input projection layers
        self.node_encoder = nn.Linear(node_feature_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_feature_dim, hidden_dim)
        
        # Graph convolution layers (using Graph Attention Networks)
        self.gnn_layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            self.gnn_layers.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=8,                    # Multi-head attention
                    dropout=0.1,
                    edge_dim=hidden_dim,        # Include edge features
                    concat=False                # Average attention heads
                )
            )
        
        # Normalization and activation
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_gnn_layers)])
        self.dropout = nn.Dropout(0.1)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Global graph embedding (for overall chip state)
        self.global_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Concat mean and max pooling
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, batch_data):
        """
        Forward pass through the GNN
        
        Args:
            batch_data: PyTorch Geometric batch containing:
                - x: node features [num_nodes, node_feature_dim]
                - edge_index: edge connections [2, num_edges] 
                - edge_attr: edge features [num_edges, edge_feature_dim]
                - batch: batch assignment [num_nodes]
        
        Returns:
            dict containing:
                - node_embeddings: [num_nodes, output_dim]
                - graph_embedding: [batch_size, output_dim]
        """
        x = batch_data.x           # Node features
        edge_index = batch_data.edge_index  # Edge connections
        edge_attr = batch_data.edge_attr    # Edge features
        batch = batch_data.batch   # Batch assignment for multiple graphs
        
        # Encode input features
        x = self.node_encoder(x)                    # [num_nodes, hidden_dim]
        edge_attr = self.edge_encoder(edge_attr)    # [num_edges, hidden_dim]
        
        # Apply graph convolution layers
        for i, gnn_layer in enumerate(self.gnn_layers):
            x_residual = x
            
            # Graph attention convolution
            x = gnn_layer(x, edge_index, edge_attr)
            
            # Layer normalization and residual connection
            x = self.layer_norms[i](x + x_residual)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Node-level embeddings (for selecting which component to place)
        node_embeddings = self.output_proj(x)       # [num_nodes, output_dim]
        
        # Graph-level e2mbedding (for understanding overall chip state)
        graph_mean = global_mean_pool(x, batch)     # [batch_size, hidden_dim]
        graph_max = global_max_pool(x, batch)       # [batch_size, hidden_dim]
        graph_embedding = self.global_proj(torch.cat([graph_mean, graph_max], dim=1))
        
        return {
            'node_embeddings': node_embeddings,
            'graph_embedding': graph_embedding
        }

class ChipDesignGraph:
    """
    Utility class to convert chip netlists into PyTorch Geometric graphs
    """
    
    def __init__(self):
        pass
    
    def create_sample_design(self, num_components=50, num_nets=80):
        """
        Create a sample chip design for testing
        
        Returns:
            PyTorch Geometric Data object
        """
        
        # Create random node features
        # [component_width, component_height, component_type, is_placed, x_pos, y_pos, is_critical, power]
        node_features = []
        
        for i in range(num_components):
            width = np.random.uniform(1, 10)        # Component width
            height = np.random.uniform(1, 10)       # Component height  
            comp_type = np.random.randint(0, 3)     # 0=gate, 1=macro, 2=IO
            is_placed = 0                           # Not placed yet
            x_pos = 0                               # No position yet
            y_pos = 0                               # No position yet
            is_critical = np.random.bernoulli(0.2)  # 20% are on critical timing paths
            power = np.random.uniform(0.1, 5.0)     # Power consumption
            
            node_features.append([width, height, comp_type, is_placed, x_pos, y_pos, is_critical, power])
        
        node_features = torch.tensor(node_features, dtype=torch.float32)
        
        # Create random edge connections (nets)
        edge_connections = []
        edge_features = []
        
        for _ in range(num_nets):
            # Each net connects 2-4 components
            net_size = np.random.randint(2, 5)
            components = np.random.choice(num_components, size=net_size, replace=False)
            
            # Create edges between all pairs in the net (complete subgraph)
            for i in range(len(components)):
                for j in range(i+1, len(components)):
                    edge_connections.append([components[i], components[j]])
                    edge_connections.append([components[j], components[i]])  # Undirected
                    
                    # Edge features: [net_weight, timing_criticality, resistance, capacitance]
                    net_weight = np.random.uniform(0.5, 2.0)
                    timing_crit = np.random.uniform(0, 1)
                    resistance = np.random.uniform(0.1, 1.0)
                    capacitance = np.random.uniform(0.1, 1.0)
                    
                    edge_features.extend([[net_weight, timing_crit, resistance, capacitance]] * 2)
        
        edge_index = torch.tensor(edge_connections, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_components
        )
        
        return data

#function test
    def fun_test():
        test_case1 = 1
        test_case = 2 
        test_case =3
        test_case 
def test_gnn():
    """
    Test the GNN on a sample chip design
    """
    print("Creating sample chip design...")
    
    # Create sample data
    graph_builder = ChipDesignGraph()
    sample_design = graph_builder.create_sample_design(num_components=100, num_nets=150)
    
    print(f"Created design with {sample_design.num_nodes} components and {sample_design.edge_index.shape[1]} connections")
    
    # Initialize GNN
    gnn = ChipPlacementGNN(
        node_feature_dim=8,
        edge_feature_dim=4, 
        hidden_dim=128,
        num_gnn_layers=4,
        output_dim=64
    )
    
    print(f"GNN has {sum(p.numel() for p in gnn.parameters())} parameters")
    
    # Forward pass
    print("Running forward pass...")
    with torch.no_grad():
        # Create batch (single design for now)
        batch_data = Batch.from_data_list([sample_design])
        
        # Forward pass
        output = gnn(batch_data)
        
        node_embeddings = output['node_embeddings']    # [100, 64]
        graph_embedding = output['graph_embedding']    # [1, 64]
        
        print(f"Node embeddings shape: {node_embeddings.shape}")
        print(f"Graph embedding shape: {graph_embedding.shape}")
        
        # These embeddings would be used by the RL policy to make placement decisions
        print("✅ GNN forward pass successful!")
        
        # Show some statistics
        print(f"Node embedding mean: {node_embeddings.mean():.4f}")
        print(f"Node embedding std: {node_embeddings.std():.4f}")
        print(f"Graph embedding norm: {torch.norm(graph_embedding):.4f}")

if __name__ == "__main__":
    test_gnn()



def test_run(g):
    for i in range(numz):
        if numz:
            

#gnn_new

        
            


