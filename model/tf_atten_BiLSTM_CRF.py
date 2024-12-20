from model.atten_submodules import *
from model.submodels import *
from logger_file import fetch_logger
import itertools
# nlpaueb/legal-bert-base-uncased

logger = fetch_logger("model file logging")
from transformers import AutoTokenizer, AutoModel

'''
    Top-level module which uses a Hierarchical-LSTM-CRF to classify.

    If pretrained = False, each example is represented as a sequence of sentences, which themselves are sequences of word tokens. Individual sentences are passed to LSTM_Sentence_Encoder to generate sentence embeddings. 
    If pretrained = True, each example is represented as a sequence of fixed-length pre-trained sentence embeddings.

    Sentence embeddings are then passed to LSTM_Emitter to generate emission scores, and finally CRF is used to obtain optimal tag sequence. 
    Emission scores are fed to the CRF to generate optimal tag sequence.
'''


class Tf_Attn_Hier_LSTM_CRF_Classifier(nn.Module):
    def __init__(self, n_tags, sent_emb_dim, sos_tag_idx, eos_tag_idx, pad_tag_idx, vocab_size=0, word_emb_dim=0,
                 pad_word_idx=0, pretrained=False, device='cuda', attention_heads = 4, num_blocks = 1):
        super().__init__()
        
        self.emb_dim = sent_emb_dim
        self.pretrained = pretrained
        self.device = device
        self.pad_tag_idx = pad_tag_idx
        self.pad_word_idx = pad_word_idx
        attention_type = "scaled_dot"
        self.attention_heads = attention_heads
        self.num_blocks = num_blocks
        self.dropout_rate = 0.2
        self.feed_forward_hidden_dim = 256
        # self.fc_attention = nn.Linear(200, 768)

        # self.encoder = TF_Encoder(self.pad_word_idx).to(self.device)
        self.encoder = AutoModel.from_pretrained('nlpaueb/legal-bert-small-uncased')
        # self.self_attention_encoder = AutoModel.from_pretrained('bert-base-uncased', output_hidden_states = True)

        # multi_headed_attention_weights = list(self.encoder.named_parameters())
        # multi_headed_attention_weights = multi_headed_attention_weights[181:197]
        
        
        
        ################################################
        multi_headed_attention_weights = []
        count = 0
        for name, param in (self.encoder).named_parameters():
            # if
            count = count + 1
            if count <= 84:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # for name, param in (self.self_attention_encoder).named_parameters():
        #     param.requires_grad = False



        for i in range(self.num_blocks):
            self.__setattr__('multihead_attn_{}'.format(i), MultiHeadAttention(model_dim=self.emb_dim,
                                                                               num_heads=self.attention_heads,
                                                                               dropout_rate=self.dropout_rate,
                                                                               attention_type=attention_type,
                                                                               query_key_value_weights = multi_headed_attention_weights,
                                                                               device=self.device))

            # self.__setattr__('feedforward_{}'.format(i), FeedForward(model_dim=sent_emb_dim,
            #                                                          hidden_dim=self.feed_forward_hidden_dim,
            #                                                          dropout_rate=self.dropout_rate,
            #                                                          query_key_value_weights = multi_headed_attention_weights))
        ######################################

        self.emitter = LSTM_Emitter(n_tags, self.emb_dim, self.emb_dim).to(self.device)
        self.crf = CRF(n_tags, sos_tag_idx, eos_tag_idx, pad_tag_idx).to(self.device)

    def forward(self, x, y=[], validation = False):
        
        batch_size = len(x)
        seq_lengths = [len(doc) for doc in x]
        max_seq_len = max(seq_lengths)

        if not self.pretrained:  ## x: list[batch_size, sents_per_doc, words_per_sent]
            # tensor_x = self.encoder(x)
            tensor_x = []
            for doc in x:
                
        
                sents = [torch.tensor(s, dtype=torch.long) for s in doc]
                sent_lengths = [len(s) for s in doc]

                ## list[sents_per_doc, words_per_sent] --> tensor[sents_per_doc, max_sent_len]
                sents = nn.utils.rnn.pad_sequence(sents, batch_first=True, padding_value=self.pad_word_idx).to(
                    self.device)
                
                hidden_reps = self.encoder(sents)
                hidden = hidden_reps[0][:, 0, :self.emb_dim]
                tensor_x.append(hidden)

        else:  ## x: list[batch_size, sents_per_doc, sent_emb_dim]
            tensor_x = [torch.tensor(doc, dtype=torch.float, requires_grad=True) for doc in x]
        
        
        tensor_x = nn.utils.rnn.pad_sequence(tensor_x, batch_first=True).to(self.device)

        # if self.pretrained:
        #   tensor_x = self.fc_attention(tensor_x).to(self.device)
          # tensor_x = torch.nn.functional.pad(input=tensor_x, pad=(0,568, 0, 0,0, 0), mode='constant', value=0)
          


        # output = self.self_attention_encoder(inputs_embeds = tensor_x)
        # all_outputs = torch.stack(output[2], dim=0).permute(2,0,1,3)

        # token_vecs_sum = []

        # for token in all_outputs:

        #     # `token` is a [13, 8, 768] tensor

        #     # Sum the vectors from the last four layers.
        #     sum_vec = torch.sum(token[-4:], dim=0)
            
        #     # Use `sum_vec` to represent `token`.
        #     token_vecs_sum.append(sum_vec)
        
        # tensor_x = torch.stack(token_vecs_sum).permute(1,0,2).to(self.device)
        # print(tensor_x.shape)

        ######### added attention on the ouptput ##################
      
        y_list = list(zip(*itertools.zip_longest(*y, fillvalue=0)))
        y_tensor = torch.as_tensor(y_list)

        if validation:
          attn_mask = attention_padding_mask(tensor_x, y_tensor, padding_index=0)  # (B, T, T)
        else:
          attn_mask = attention_padding_mask_efl(tensor_x, y_tensor, padding_index=0)  # (B, T, T)

        attn_mask = attn_mask.to(self.device)
        
        for i in range(self.num_blocks):
            tensor_x, _ = self.__getattr__('multihead_attn_{}'.format(i))(tensor_x, tensor_x, tensor_x,
                                                                          attn_mask=attn_mask)  # (B, T, D)

            #tensor_x = self.__getattr__('feedforward_{}'.format(i))(tensor_x)  # (B, T, D)

        self.mask = torch.zeros(batch_size, max_seq_len).to(self.device)
        for i, sl in enumerate(seq_lengths):
            self.mask[i, :sl] = 1

        ########## added attention on the output ################
        #tensor_x = self.fc_encoder_output(tensor_x)
        self.emissions = self.emitter(tensor_x)
      
        _, path = self.crf.decode(self.emissions, mask=self.mask)
        return path, self.emissions

    def _loss(self, y):
        ##  list[batch_size, sents_per_doc] --> tensor[batch_size, max_seq_len]
        tensor_y = [torch.tensor(doc, dtype=torch.long) for doc in y]
        tensor_y = nn.utils.rnn.pad_sequence(tensor_y, batch_first=True, padding_value=self.pad_tag_idx).to(self.device)

        nll = self.crf(self.emissions, tensor_y, mask=self.mask)
        return nll
