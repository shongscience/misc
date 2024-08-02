import torch
import torch.nn as nn
import torch.nn.functional as F
from .odst import ODST


class DenseBlock(nn.Sequential):
    def __init__(self, input_dim, layer_dim, num_layers, tree_dim=1, max_features=None,
                 input_dropout=0.0, flatten_output=True, Module=ODST, **kwargs):
        layers = []
        for i in range(num_layers):
            oddt = Module(input_dim, layer_dim, tree_dim=tree_dim, flatten_output=True, **kwargs)
            input_dim = min(input_dim + layer_dim * tree_dim, max_features or float('inf'))
            layers.append(oddt)

        super().__init__(*layers)
        self.num_layers, self.layer_dim, self.tree_dim = num_layers, layer_dim, tree_dim
        self.max_features, self.flatten_output = max_features, flatten_output
        self.input_dropout = input_dropout

    def forward(self, x):
        #print("Entering DenseBlock forward pass")
        initial_features = x.shape[-1]
        #print("initial_features=",initial_features)
        
        #for layer in self:
        #print("Layers, Max_Features: ",self.num_layers,self.max_features)
        for i, layer in enumerate(self):
            #print(f"Layer {i+1}/{self.num_layers}")
            layer_inp = x
            #print(f"Layer input shape: {layer_inp.shape}")
    
            if self.max_features is not None:
                tail_features = min(self.max_features, layer_inp.shape[-1]) - initial_features
                if tail_features != 0:
                    layer_inp = torch.cat([layer_inp[..., :initial_features], layer_inp[..., -tail_features:]], dim=-1)
                #print(f"After max_features adjustment, layer input shape: {layer_inp.shape}")
    
            if self.training and self.input_dropout:
                layer_inp = F.dropout(layer_inp, self.input_dropout)
                #print(f"Applied dropout, layer input shape: {layer_inp.shape}")
    
            h = layer(layer_inp)
            #print(f"Layer output shape: {h.shape}")
    
            x = torch.cat([x, h], dim=-1)
            #print(f"Concatenated output shape: {x.shape}")
            #print(f"Layer {i+1} output shape: {x.shape}")

        outputs = x[..., initial_features:]
        if not self.flatten_output:
            outputs = outputs.view(*outputs.shape[:-1], self.num_layers * self.layer_dim, self.tree_dim)
            
        #print("Exiting DenseBlock forward pass")
        return outputs