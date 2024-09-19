# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 20:09:39 2024

@author: hazem
"""

#General



#pandas, numpy and sklearn



#pytorch tools
import torch
import torch.nn as nn
import torch_geometric
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import torch.optim as optim
from torch_geometric.utils import to_networkx
from torch_geometric.data import Dataset, Data, download_url
from torch_geometric.loader import DataLoader
#from torch_geometric.data import DataLoader
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.nn import GATConv,HeteroConv
from datasets import list_datasets, load_dataset, list_metrics, load_metric
import nltk


class HeteroGATFlexSentenceModel(nn.Module):
    def __init__(self, num_node_features, hidden_word_dim, hidden_sen_dim, num_heads, num_classes=2, dropout_rate=0.3):
        super(HeteroGATFlexSentenceModel, self).__init__()

        # Define the HeteroConv layers
        self.conv1 = HeteroConv({
            ('Words', 'belong', 'Sentences'): GATConv(
                in_channels=(num_node_features['Words'], num_node_features['Sentences']),
                out_channels=hidden_word_dim[0],
                heads=num_heads['Words'],
                add_self_loops=False
            ),
            ('Words', 'similar', 'Words'): GATConv(
                in_channels=(num_node_features['Words'], num_node_features['Words']),
                out_channels=hidden_word_dim[1],
                heads=num_heads['Words'],
                add_self_loops=False
            ),
            ('Sentences', 'Similarity', 'Sentences'): GATConv(
                in_channels=(num_node_features['Sentences'], num_node_features['Sentences']),
                out_channels=hidden_sen_dim[0],
                heads=num_heads['Sentences'],
                add_self_loops=False
            ),
            ('Sentences', 'Adjacency', 'Sentences'): GATConv(
                in_channels=(hidden_sen_dim[0] * num_heads['Sentences'], hidden_sen_dim[0] * num_heads['Sentences']),
                out_channels=hidden_sen_dim[1],
                heads=num_heads['Sentences'],
                add_self_loops=False
            ),
            ('Sentences', 'Contain', 'Words'): GATConv(
                in_channels=(hidden_sen_dim[1] * num_heads['Sentences'], hidden_word_dim[1]),
                out_channels=hidden_word_dim[2],
                heads=num_heads['Sentences'],
                add_self_loops=False
            ),
        }, aggr='sum')
        

        self.dropout = nn.Dropout(p=dropout_rate)

        # Classification head output layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_sen_dim[2] * num_heads['Sentences'], num_classes),  # Adjusted to match hidden_sen_dim[1]
            nn.Sigmoid()
        )

    def forward(self, x_dict, edge_index_dict):
        print("Input x_dict:", x_dict)
        print("Input edge_index_dict:", edge_index_dict)

        # Apply the HeteroConv layer
        x_dict = self.conv1(x_dict, edge_index_dict)
        print("Output after conv1:", x_dict)

        # Apply ReLU and dropout
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        print("x_dict after ReLU:", x_dict)
        
        if 'Sentences' in x_dict:
            x_dict['Sentences'] = self.dropout(x_dict['Sentences'])
            print("x_dict['Sentences'] after dropout:", x_dict['Sentences'])
        else:
            print("Key 'Sentences' not found in x_dict after dropout")

        # Apply the classification layer on the sentence node features
        if 'Sentences' in x_dict:
            out = self.fc(x_dict['Sentences'])
            print("Output after classification layer:", out)
        else:
            print("Key 'Sentences' not found in x_dict for classification")
            out = None

        return out