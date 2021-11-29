import torch
import torch_geometric.nn as pyg_nn



class GNNmodel(torch.nn.Module):
    def __init__(self, av, input_dim, output_dim):
        """
            Generic GNN model
        """
        super(GNNmodel, self).__init__()
        self.av = av
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.build_conv_layers()
   

    def build_conv_layers(self):
        if self.av.CONV == "Graph":  
          conv_layer = pyg_nn.GraphConv
        elif self.av.CONV == "GCN":  
          conv_layer = pyg_nn.GCNConv
        elif self.av.CONV == "GAT":
          conv_layer = pyg_nn.GATConv
        elif self.av.CONV == "SAGE":
          conv_layer = pyg_nn.SAGEConv
        elif self.av.CONV == "GIN":
          conv_layer = lambda i, h: pyg_nn.GINConv(nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h)))            
        else: 
          raise NotImplementedError()  
        #Conv layers
        self.conv1 = conv_layer(self.input_dim, self.av.filters_1)
        self.conv2 = conv_layer(self.av.filters_1, self.av.filters_2)
        self.conv3 = conv_layer(self.av.filters_2, self.av.filters_3)
        self.fc1 = torch.nn.Linear(self.av.filters_3,self.av.filters_3)
        self.fc2 = torch.nn.Linear(self.av.filters_3, self.output_dim)
        self.readout = torch.nn.LogSoftmax(dim=-1)

    def GNN (self, data):
        """
        """
        features = self.conv1(data.x,data.edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, p=self.av.DROPOUT, training=self.training)

        features = self.conv2(features,data.edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, p=self.av.DROPOUT, training=self.training)

        features = self.conv3(features,data.edge_index)
        return features
    

    def forward(self, batch_data):
        """
        """
        gnn_node_level_feats = self.GNN(batch_data)
        gnn_graph_level_feats = pyg_nn.global_add_pool(gnn_node_level_feats, batch_data.batch)
        transformed_feats = torch.nn.functional.relu(self.fc1(gnn_graph_level_feats))
        scores = self.fc2(torch.nn.functional.dropout(transformed_feats, p=self.av.DROPOUT, training=self.training))
        return self.readout(scores)

