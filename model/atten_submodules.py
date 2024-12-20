import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class FeedForward(nn.Module):

    def __init__(self, model_dim=512, hidden_dim=2048, dropout_rate=0.0, query_key_value_weights = []):
        super().__init__()

        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate


        self.linear1 = nn.Linear(self.model_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.model_dim)
        self.norm = nn.LayerNorm(self.model_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # with torch.no_grad():
        #   print(self.linear1.weight.shape)
        #   print(query_key_value_weights[10][1].shape)

        #   self.linear1.weight.copy_(query_key_value_weights[10][1][:, :200])
        #   self.linear1.bias.copy_(query_key_value_weights[11][1])

        #   self.linear2.weight.copy_(query_key_value_weights[12][1][:200, :])
        #   self.linear2.bias.copy_(query_key_value_weights[13][1][:200])

        #   self.norm.weight.copy_(query_key_value_weights[14][1][:200])
        #   self.norm.bias.copy_(query_key_value_weights[15][1][:200])




    def forward(self, x):
        output = self.linear2(F.relu(self.linear1(x)))

        output = self.dropout(output)

        output = self.norm(output + x)
        return output


def attention_padding_mask_efl(q, k, padding_index=0):


  if k.shape[0]!=1:
    print('batch size should be 1')
    sys.exit(1)

  else:

    q = torch.mean(q,2)

    mask = k.eq(padding_index).unsqueeze(1).expand(-1, q.size(-1), -1)
    labels = k.squeeze(dim=0)

    for itr, sent_label in enumerate(labels):
      if sent_label.item() == 3:
        mask[0][itr] = torch.BoolTensor([True]*mask.shape[-1]).to('cuda')
      else:
        mask[0][itr] = torch.BoolTensor([False]*mask.shape[-1]).to('cuda')

    return mask.to('cuda')


def attention_padding_mask(q, k, padding_index=0):
    """Generate mask tensor for padding value
    Args:
        q (Tensor): (B, T_q)
        k (Tensor): (B, T_k)
        padding_index (int): padding index. Default: 0
    Returns:
        (torch.BoolTensor): Mask with shape (B, T_q, T_k). True element stands for requiring making.
    Notes:
        Assume padding_index is 0:
        k.eq(0) -> BoolTensor (B, T_k)
        k.eq(0).unsqueeze(1)  -> (B, 1, T_k)
        k.eq(0).unsqueeze(1).expand(-1, q.size(-1), -1) -> (B, T_q, T_k)
    """
    ## we take the mean because we want to get rid of last dim.
    ### what we do to remove that dim deosn't matter, since we are only ending up with
    ### true/false for mask.

    q = torch.mean(q,2)

    mask = k.eq(padding_index).unsqueeze(1).expand(-1, q.size(-1), -1)
    return mask


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention calculation"""

    def __init__(self, num_heads = 2, dropout_rate=0.0, **kwargs):
        """Initialize ScaledDotProductAttention
        Args:
            dropout_rate (float): attention dropout_rate rate
        """
        super().__init__()
        print("inside scaled dot prod attention")

        self.dropout = nn.Dropout(dropout_rate)
        # self.merged_heads = nn.Linear(num_heads, 1)




    def forward(self, Q, K, V, attn_mask = None):

      scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(K.size(-1)) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
      scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
      attn = nn.Softmax(dim=-1)(scores)
      attn = self.dropout(attn)
      context = torch.matmul(attn, V)
      
      # attention_weights = self.merged_heads(attn.permute(0,2,3,1))
      return context, attn

    # def forward(self, q, k, v, attn_mask=None):
    #     """Forward
    #     Args:
    #         q (torch.Tensor): Query matrix, (B, T_q, D_q)
    #         k (torch.Tensor): Key matrix, (B, T_k, D_k)
    #         v (torch.Tensor): Value matrix, (B, T_v, D_v) T_v = T_k, D_v = D_k
    #         attn_mask (torch.BoolTensor | None): Mask tensor. True element will be masked.
    #     Returns:
    #         output (B, T_q, D_v); attention (B, T_q, T_k)
    #     """
    #     attention = torch.bmm(q, k.permute(0, 2, 1))  # (B, T_q, T_k)

    #     # Scale
    #     attention *= k.size(-1) ** -0.5

    #     if attn_mask is not None:
    #         attention.masked_fill_(attn_mask, -np.inf)  # positions that require masking are now -np.inf

    #     attention = F.softmax(attention, dim=-1)

    #     attention = self.dropout(attention)

    #     output = attention.bmm(v)  # (B, T_q, D_v)

    #     return output, attention


class CosineAttention(nn.Module):
    """Cosine Attention"""

    def __init__(self, dropout_rate=0.0, eps=1e-10, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.eps = eps

    def forward(self, q, k, v, attn_mask=None):
        """Forward
        Args:
            q (torch.Tensor): Query matrix, (B, T_q, D_q)
            k (torch.Tensor): Key matrix, (B, T_k, D_k)
            v (torch.Tensor): Value matrix, (B, T_v, D_v) T_v = T_k, D_v = D_k
            attn_mask (torch.BoolTensor | None): Mask tensor. True element will be masked.
        Returns:
            output (B, T_q, D_v); attention (B, T_q, T_k)
        Notes:
            Consine attention requires D_q = D_k, so I denote it as D here
        """

        q_norm = q / (q.norm(p=2, dim=-1, keepdim=True) + self.eps)  # (B, T_q, D)
        k_norm = k / (k.norm(p=2, dim=-1, keepdim=True) + self.eps)  # (B, T_k, D)

        attention = torch.bmm(q_norm, k_norm.permute(0, 2, 1))  # (B, T_q, T_k)

        if attn_mask is not None:
            attention.masked_fill_(attn_mask, -np.inf)  # positions that require masking are now -np.inf

        attention = F.softmax(attention, dim=-1)

        attention = self.dropout(attention)

        output = attention.bmm(v)  # (B, T_q, D_v)

        return output, attention


class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=512, num_heads=4, dropout_rate=0.0, attention_type='scaled_dot', query_key_value_weights = [], device = "cuda"):
        super().__init__()
        assert model_dim % num_heads == 0, 'model_dim should be devided by num_heads'
        self.h_size = model_dim
        self.num_heads = num_heads
        self.device = device
        self.head_h_size = model_dim // num_heads

        self.linear_q = nn.Linear(self.h_size, self.h_size)
        self.linear_k = nn.Linear(self.h_size, self.h_size)
        self.linear_v = nn.Linear(self.h_size, self.h_size)
        self.fc0 = nn.Linear(self.h_size, self.h_size)

        ## positional encoding to be added
        # self.positional_encoder = PositionalEncoding(self.h_size).to(self.device)
        ### newly added dropout
        self.attention = ScaledDotProductAttention(dropout_rate = 0.2).to(self.device)
        # self.attention = CosineAttention(dropout_rate = 0.2)
        self.dropout = nn.Dropout(dropout_rate)
        self.lnorm = nn.LayerNorm(model_dim)

        
        # with torch.no_grad():
          # self.linear_q.weight.copy_(query_key_value_weights[0][1][:200, :200])
          # self.linear_q.weight.requires_grad = False

          # self.linear_q.bias.copy_(query_key_value_weights[1][1][:200])
          # self.linear_q.bias.requires_grad = False

          # self.linear_k.weight.copy_(query_key_value_weights[2][1][:200, :200])
          # self.linear_k.weight.requires_grad = False

          # self.linear_k.bias.copy_(query_key_value_weights[3][1][:200])
          # self.linear_k.bias.requires_grad = False


          # self.linear_v.weight.copy_(query_key_value_weights[4][1][:200, :200])
          # self.linear_v.weight.requires_grad = False

          # self.linear_v.bias.copy_(query_key_value_weights[5][1][:200])
          # self.linear_v.bias.requires_grad = False

          # self.fc0.weight.copy_(query_key_value_weights[6][1][:200 , :200])
          # self.fc0.bias.copy_(query_key_value_weights[7][1][:200])

          # self.lnorm.weight.copy_(query_key_value_weights[8][1][:200])
          # self.lnorm.bias.copy_(query_key_value_weights[9][1][:200])
          # self.lnorm.weight.requires_grad = False
          # self.lnorm.bias.requires_grad = False



          # self.linear_v.bias.requires_grad = False   
    # def add_positional_mask (self, q, k, v):
    #   q = self.positional_encoder(q)
    #   k = self.positional_encoder(k)
    #   v = self.positional_encoder(v)

    #   return q, k, v


    def forward(self, q, k, v, attn_mask=None):
        batch_size = q.size(0)

        tensor1 = []
      

        # Residual
        residual = q

        # q, k, v = self.add_positional_mask(q, k, v)

        # Linear projection
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        # Form multi heads
        q = q.view(batch_size, -1, self.num_heads, self.head_h_size).transpose(1,2)  # (h * B, T_q, D / h)
        k = k.view(batch_size, -1, self.num_heads, self.head_h_size).transpose(1,2)  # (h * B, T_k, D / h)
        v = v.view(batch_size, -1, self.num_heads, self.head_h_size).transpose(1,2)  # (h * B, T_v, D / h)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).to(self.device)  # (h * B, T_q, T_k)

       
        
        context, attention_per_head = self.attention(q, k, v, attn_mask=attn_mask)
        # context: (h * B, T_q, D_v) attention: (h * B, T_q, T_k)

        # Concatenate heads
        context = context.view(batch_size, -1, self.h_size)  # (B, T_q, D)

        
        # Dropout
        output = self.dropout(self.fc0(context))  # (B, T_q, D)

        # Residual connection and Layer Normalization
        output = self.lnorm(residual + output)  # (B, T_q, D)

        return output, attention_per_head


        

def decode_entity(x, mask):
    """Decode sequences of entities from weight matrix
    Args:
        x (torch.Tensor): output with shape (B, T, num_entities)
        mask (torch.BoolTensor): (B, T)
    Returns:
        (list[list[int]]): best sequences of entities of this batch, representing in indexes (B, *)
    """
    first_invalid = mask.sum(1)  # (B,)

    preds = x.argmax(dim=-1)  # (B, T)
    path = [preds[i].data[:first_invalid[i].item()].tolist() for i in range(preds.shape[0])]  # (B, *)
    return path