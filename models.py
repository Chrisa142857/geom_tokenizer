
import transformers
import torch

class BERT(torch.nn.Module):
    def __init__(self, node_num, node_channel, geom_dim, cls_num, dropout=0.1, hdim=2048, nhead=8, pe_dim=0) -> None:
        '''
        Toy transformer for node classification
        '''
        super().__init__()
        ## Embed all tokens
        # self.encoder = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.encoder = transformers.BertModel(transformers.BertConfig(hidden_size=hdim, num_attention_heads=nhead))
        # self.encoder.config.output_attentions = True
        hdim = self.encoder.config.hidden_size
        tokens_num = 1
        # token_embed1 = torch.nn.Embedding(node_num, hdim//tokens_num)
        token_embed2 = torch.nn.Embedding(2**geom_dim, hdim//tokens_num)
        token_embed3 = torch.nn.Linear(node_channel, hdim//tokens_num)
        # token_embed3 = torch.nn.Embedding(node_num**2, hdim//tokens_num)
        # token_embed4 = torch.nn.Embedding(node_num, hdim//4)
        self.token_embeds = torch.nn.ModuleList([token_embed2, token_embed3]) #, token_embed4
        self.node_embed =  torch.nn.Linear(node_channel+pe_dim, hdim) #, token_embed4
        ## Transformer Encoder
        # encoder_layer = torch.nn.TransformerEncoderLayer(d_model=hdim, nhead=nhead)
        # self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.classifier = torch.nn.Linear(hdim, cls_num)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, inputs, masks=None):
        # x, pos_tokens, geom_tokens, view_tokens = inputs
        ## geom_token + view_token
        embeds = []
        for f, token in zip(self.token_embeds, inputs[1:]):
           embeds.append(f(token))
        embeds = torch.stack(embeds, 0).sum(0)
        ## geom_token + view_token + node feat
        embeds = embeds + self.node_embed(inputs[0])
        embeds = self.dropout(embeds)
        ## node feat
        # embeds = self.node_embed(inputs[0])
        outputs = self.encoder(inputs_embeds=embeds, attention_mask=masks)
        ## last_hidden_state, pooler_output, attentions = outputs
        out = self.classifier(outputs[1])
        return out