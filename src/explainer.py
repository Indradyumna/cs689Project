import torch
import torch_geometric.nn as pyg_nn
import math
from src.utils import cudavar


class GNNexplainer(torch.nn.Module):
    def __init__(self, av,model,num_nodes,num_feats,num_edges):
        """
        """
        super(GNNexplainer, self).__init__()
        self.av = av
        self.model = model
        self.num_nodes = num_nodes
        self.feat_mask_len = num_feats
        self.edge_mask_len = num_edges
        self.gnn_models_list = list(filter(lambda m: isinstance(m, pyg_nn.MessagePassing), self.model.modules()))
        self.init_mask()
        
    def init_mask(self):
        self.node_feat_mask = torch.nn.Parameter(torch.FloatTensor(self.feat_mask_len))
        torch.nn.init.normal_(self.node_feat_mask,0.1,0.1)
        
        #init strategy taken from 
        #https://github.com/RexYing/gnn-model-explainer/blob/master/explainer/explain.py#L648
        std = torch.nn.init.calculate_gain("relu") * math.sqrt(
                2.0 / (self.num_nodes + self.num_nodes)
            )
        self.edge_mask = torch.nn.Parameter(cudavar(self.av,torch.randn(self.edge_mask_len)) * std)

        #Leveraging inbuilt support for GNNexplainer in pytorch_geometric
        #Refer to docs here : 
        #https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/message_passing.html
        for m in self.gnn_models_list:
            m.__explain__ = True
            m.__edge_mask__ = self.edge_mask        
        
    def clear_mask(self):
        #reset changes to module when explanation task is done 
        for m in self.gnn_models_list:
            m.__explain__ = False
            m.__edge_mask__ = None
    
    def forward(self,data):
                
        data.x = data.x * cudavar(self.av,torch.sigmoid(self.node_feat_mask.unsqueeze(0)))

        #Our gnn outputs LogSoftmax probabilities
        loss = -self.model(data)[:, data.y]
        
        #sigmoid activation of masks
        feat_mask = torch.sigmoid(self.node_feat_mask)
        edge_mask = torch.sigmoid(self.edge_mask)

        # mask size losses
        loss = loss + self.av.FEAT_MASK_SZ_COEFF * torch.mean(feat_mask)
        loss = loss + self.av.EDGE_MASK_SZ_COEFF * torch.mean(edge_mask)

        # mask entropy loss.
        feat_mask_ent = -feat_mask * torch.log(feat_mask) - (1 - feat_mask) * torch.log(1 - feat_mask)
        edge_mask_ent = -edge_mask * torch.log(edge_mask) - (1 - edge_mask) * torch.log(1 - edge_mask)
        
        loss = loss + self.av.FEAT_MASK_ENT_COEFF * torch.mean(feat_mask_ent)
        loss = loss + self.av.EDGE_MASK_ENT_COEFF * torch.mean(edge_mask_ent)

        return loss.sum()       

