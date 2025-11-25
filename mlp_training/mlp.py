import torch

class MLP(torch.nn.Module):
    def __init__(self, num_tgt_model_layers, layer_dims, dropout=0.15, mode='lin_agt'):
        super(MLP, self).__init__()
        self.mode = mode
        self.layer_dims = layer_dims
        self.num_tgt_model_layers = num_tgt_model_layers

        # learnable linear weighting for each layer
        self.layer_weights = torch.nn.Parameter(torch.ones(self.num_tgt_model_layers))

        layers_ls = [[
            torch.nn.Linear(layer_dims[i], layer_dims[i+1]),
            torch.nn.BatchNorm1d(layer_dims[i+1]),
            torch.nn.GELU(),
            torch.nn.Dropout(p=dropout)
        ] for i in range(len(layer_dims)-1)]

        # everything after the weighted-sum operation
        self.model = torch.nn.Sequential(
            *[l for layers in layers_ls for l in layers][:-3],  # excluding last 3 (BN, GELU, Dropout) if needed
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch_size, num_tgt_model_layers, layer_dims[0])
        # apply learnable weights to each layer
        if self.mode == 'lin_agt':
            current_weights = self.layer_weights.to(x.device)
            weights = torch.softmax(current_weights, dim=0)     # normalize across layers
            x = (x * weights.view(1, -1, 1)).sum(dim=1)         # shape -> (batch_size, 4096)
        
        elif self.mode == 'avg_agt':
            num_layers = x.shape[1]
            x = x.sum(dim=1)/num_layers  # simple sum across layers
        
        else:
            layer_num = self.mode.split('_')[-1]
            x = x[:, int(layer_num), :]  # select specific layer

        return self.model(x)

    def set_dropout(self, dropout):
        for module in self.model:
            if isinstance(module, torch.nn.Dropout):
                module.p = dropout

    def set_mode(self, mode):
        if mode not in ['lin_agt', 'avg_agt'] and not mode.startswith('lyr_'): 
            raise ValueError("Invalid mode. Choose from 'lin_agt', 'avg_agt', or 'lyr_{layer_number}'")
        self.mode = mode