import torch
from torch import nn
import torch_geometric.nn as geom_nn

class MLPModel(nn.Module):
    def __init__(self, c_in, c_out, c_hidden, num_layers=2, dp_rate=0.1):
        """
        c_in - Dimension of input features
        c_hidden - Dimension of hidden features
        c_out - Dimension of the output features. Usually number of classes in classification
        num_layers - Number of hidden layers
        dp_rate - Dropout rate to apply throughout the network
        """
        super().__init__()
        layers = []
        for l in range(num_layers):
            layers += [nn.Linear(c_in, c_hidden),
                      nn.ReLU(inplace = True),
                      nn.Dropout(dp_rate)]
            c_in = c_hidden
        layers += [nn.Linear(c_hidden, c_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x, *args, **kwargs):
        return self.layers(x)


class NodeLevelGNN(nn.Module):
    def __init__(self):
        pass




class GNNModel(nn.Module):
    def __init__(self, c_in, c_out, c_hidden, num_layers=2, layer_name="GCN", dp_rate=0.1, **kwargs):
        super().__init__()
        gnn_layer_by_name = {
                        "GCN": geom_nn.GCNConv,
                        "GAT": geom_nn.GATConv,
                        "GraphConv": geom_nn.GraphConv
                    }
        

        gnn_layer = gnn_layer_by_name[layer_name]
        
        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers-1):
            layers += [
                gnn_layer(in_channels=in_channels, 
                          out_channels=out_channels,
                          **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = c_hidden
        layers += [gnn_layer(in_channels=in_channels, 
                             out_channels=c_out,
                             **kwargs)]
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x, edge_index, *args, **kwargs):
        for l in self.layers:
            if isinstance(l, geom_nn.MassagePassing):
                x = l(x, edge_index)
            else:
                x = l(x)
        return x 






if __name__ == "__main__":
    c_in = 12
    c_hidden = 32
    b = 3
    c_out = 3
    input = torch.randn(3, 12)
    model = MLPModel(c_in, c_out, c_hidden)
    output = model(input)
    print("output: ", output, output.size())

