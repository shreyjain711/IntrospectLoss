import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

class MLP_V2(torch.nn.Module):
    def __init__(self, model_name, dropout=0.15, mode='lin_agt'):
        super(MLP_V2, self).__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.mode = mode
        self.layer_dims = config.n_layers
        self.num_tgt_model_layers = config.num_hidden_layers + 1

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            output_hidden_states=True,
            local_files_only=True
        )
        self.llm.eval()
        self.llm.requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # learnable linear weighting for each layer
        self.layer_weights = torch.nn.Parameter(torch.ones(self.num_tgt_model_layers))

        layers_ls = [[
            torch.nn.Linear(self.layer_dims[i], self.layer_dims[i+1]),
            torch.nn.BatchNorm1d(self.layer_dims[i+1]),
            torch.nn.GELU(),
            torch.nn.Dropout(p=dropout)
        ] for i in range(len(self.layer_dims)-1)]

        # everything after the weighted-sum operation
        self.model = torch.nn.Sequential(
            *[l for layers in layers_ls for l in layers][:-3],
            torch.nn.Sigmoid()
        )

    def forward_mlp(self, x):
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
    
    def forward(self, messages_batch):
        input_messages_batch = [self.tokenizer.apply_chat_template([messages[0]], tokenize=False) for messages in messages_batch] # messages 
        full_messages_batch = [self.tokenizer.apply_chat_template(messages, tokenize=False) for messages in messages_batch] # messages 
        inputs = self.tokenizer(full_messages_batch, return_tensors="pt", padding=True).to(self.llm.device)
        
        with torch.no_grad():
            outputs = self.llm(**inputs)
            hidden_states = outputs.hidden_states  # tuple of (layer_num, batch_size, seq_len, hidden_size)
        
        # get hidden states only from the positions of the last tokens of input messages
        last_token_indices = []
        for i, input_messages in enumerate(input_messages_batch):
            input_ids = self.tokenizer(input_messages, return_tensors="pt").input_ids
            last_token_index = (input_ids != self.tokenizer.pad_token_id).sum() - 1
            last_token_indices.append(last_token_index)
        layer_outputs = torch.stack([hs[:, :, last_token_indices, :] for hs in hidden_states], dim=1)  # (batch_size, num_layers, hidden_size)
        
        # get an mlp output from each seq position in layer_outputs
        mlp_outputs = []
        for sl in range(layer_outputs.shape[2]):
            mlp_outputs.append(self.forward_mlp(layer_outputs[:, :, sl, :]))
        
        return torch.stack(mlp_outputs, dim=1).squeeze(-1)  # (batch_size, seq_len)

    def set_dropout(self, dropout):
        for module in self.model:
            if isinstance(module, torch.nn.Dropout):
                module.p = dropout

    def set_mode(self, mode):
        if mode not in ['lin_agt', 'avg_agt'] and not mode.startswith('lyr_'): 
            raise ValueError("Invalid mode. Choose from 'lin_agt', 'avg_agt', or 'lyr_{layer_number}'")
        self.mode = mode

    def save_weights(self, path):
        checkpoint = {
            "model": self.model.state_dict(),
            "layer_weights": self.layer_weights.detach().cpu()
        }
        torch.save(checkpoint, path)

    def load_weights(self, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location)
        self.model.load_state_dict(ckpt["model"])
        self.layer_weights.data.copy_(ckpt["layer_weights"].to(self.layer_weights.device))
