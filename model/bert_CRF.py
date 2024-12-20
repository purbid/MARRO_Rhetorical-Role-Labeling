from model.submodels import *
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification


class Bert_CRF(nn.Module):
    def __init__(self, n_tags, sent_emb_dim, sos_tag_idx, eos_tag_idx, pad_tag_idx, vocab_size=0, word_emb_dim=0,
                 pad_word_idx=0, pretrained=False, device='cpu'):

        super().__init__()

        ############ from bert model ##########
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=7)


        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, n_tags)

        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)


        ######### from CRF MODEL ####################

        self.emb_dim = sent_emb_dim
        self.pretrained = pretrained
        self.device = device
        self.pad_tag_idx = pad_tag_idx
        self.pad_word_idx = pad_word_idx

        # sentence encoder is not required for pretrained embeddings
        self.crf = CRF(n_tags, sos_tag_idx, eos_tag_idx, pad_tag_idx).to(self.device)


    def tokenise_input(self, batch_input):

        return


    def forward(self, input_ids, attention_mask=None, labels=None):

        #################### bert model ###########################
        '''

        :param x: [batch_size, sentence_len, embedding_size]
        '''

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
        # print(outputs["logits"].shape)
        # return outputs
        # op1 = self.fc1(cls_hs)
        # op1 = self.relu(op1)
        # op1 = self.dropout(op1)
        # # output layer
        # op1 = self.fc2(op1)
        # # apply softmax activation
        # bert_final_op = self.softmax(op1)


        ######################## HIERAR MODEL ###########################

        # batch_size = len(x)
        # seq_lengths = [len(doc) for doc in x]
        # max_seq_len = max(seq_lengths)
        '''
        crf decoder accepts bert final output and attention mask
        '''
        ### masks will be batch size * max_Seq => 1, our_batch_size

        # 32*658*10 => 32documents*658sentencesmax*eachsentence200embedding
        # attention should be 32documents*ones upto number of sentences.
        _, path = self.crf.decode(outputs["logits"].unsqueeze(0), mask=attention_mask)
        return path

    def _loss(self, y):
        ##  list[batch_size, sents_per_doc] --> tensor[batch_size, max_seq_len]
        tensor_y = [torch.tensor(doc, dtype=torch.long) for doc in y]
        tensor_y = nn.utils.rnn.pad_sequence(tensor_y, batch_first=True, padding_value=self.pad_tag_idx).to(self.device)

        nll = self.crf(self.emissions, tensor_y, mask=self.mask)
        return nll
