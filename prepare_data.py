import string
import random
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
encoder = AutoModel.from_pretrained('nlpaueb/legal-bert-base-uncased')
'''
    This function constructs folds that have a balanced category distribution.
    Folds are stacked up together to give the order of docs in the main data.
    
    idx_order defines the order of documents in the data. Each sequence of (docs_per_fold) documents in idx_order can be treated as a single fold, containing documents balanced across each category.
'''
def prepare_folds(args):
    with open(args.cat_path) as fp:
        
        categories = []
        for line in fp:
            _, docs = line.strip().split('\t')
            docs = docs.strip().split(' ')
            categories.append(docs)

    # categories: list[category, docs_per_category]

    categories.sort(key = lambda x: len(x))
    n_docs = len(sum(categories, []))
    print(n_docs)
    assert n_docs == args.dataset_size, "invalid category list"
           
    docs_per_fold = args.dataset_size // args.num_folds   
    folds = [[] for f in range(docs_per_fold)]
    print(folds)
    
    # folds: list[num_folds, docs_per_fold]
    
    f = 0
    for cat in categories:
        for doc in cat:
            folds[f].append(doc)
            f = (f + 1) % 5

    # list[num_folds, docs_per_fold] --> list[num_folds * docs_per_fold]
    idx_order = sum(folds, [])
    return idx_order


descriptions_dict = {'FAC':'This is facts FAC',
  'RLC' : 'This is Ruling of Lower Court RLC',
  'ARG':'This is an Argument ARG',
  'STA' : 'This is Stature STA',
  'PRE' : 'This is Precedent PRE',
  'Ratio' : 'This is Ratio',
  'RPC' : 'This is present court judgement RPC'}

def add_negative_examples_and_change_dataset(document_sentence, label_y, tag2idx):
  labels = []
  negative_sentences = []

  for lab in descriptions_dict:
    if lab!=label_y:
      tokens = tokenizer.tokenize(document_sentence)
      label_description_tokens = tokenizer.tokenize(descriptions_dict[lab])
      if len(tokens) >= 50:
        tokens = tokens[0:50]
      tokens = ['[CLS]'] + tokens + ['[SEP]'] + label_description_tokens + ['[SEP]']
      sent_x = tokenizer.convert_tokens_to_ids(tokens)
      negative_sentences.append(sent_x)
      labels.append(tag2idx["entailed"])
  try:
    neg_examples = random.sample(negative_sentences, random.randint(1,6))
  except:
    print(negative_sentences)

  return labels[:len(neg_examples)], neg_examples


def prepare_data_original(idx_order, args):
    x, y = [], []

    word2idx = defaultdict(lambda: len(word2idx))
    tag2idx = defaultdict(lambda: len(tag2idx))

    # map the special symbols first
    word2idx['<pad>'], word2idx['<unk>'],word2idx['[CLS]'],word2idx['[SEP]'] = 0, 1,2,3
    tag2idx['<pad>'], tag2idx['<start>'], tag2idx['<end>'] = 0, 1, 2

    # iterate over documents
    for doc in idx_order:
        doc_x, doc_y = [], [] 

        with open(args.data_path + doc + '.txt') as fp:
            # iterate over sentences
            for sent in fp:
                #print(sent)
                try:
                	sent_x, sent_y = sent.strip().split('\t')
                except ValueError:
                	continue

                # cleanse text, map words and tags
                if not args.pretrained:
                    tokens=tokenizer.tokenize(sent_x)
                    if len(tokens)>=50:
                        tokens=tokens[0:50]
                    tokens=['[CLS]']+tokens+['[SEP]']
                    sent_x=tokenizer.convert_tokens_to_ids(tokens)
                else:
                    sent_x = list(map(float, sent_x.strip().split()[:args.emb_dim]))
                sent_y = tag2idx[sent_y.strip()]

                if sent_x != []:
                   # print(sent_x)
                    doc_x.append(sent_x)
                    doc_y.append(sent_y)
        
        x.append(doc_x)
        y.append(doc_y)

    return x, y, word2idx, tag2idx, []



def prepare_data_copy(idx_order, args):
    x, y, targets = [], [], []
    full_word_doc_x, full_word_doc_y, full_text_labels = [], [], []
    original_sentences = {}

    word2idx = defaultdict(lambda: len(word2idx))
    tag2idx = defaultdict(lambda: len(tag2idx))

    # map the special symbols first
    word2idx['<pad>'], word2idx['<unk>'], word2idx['[CLS]'], word2idx['[SEP]'] = 0, 1, 2, 3
    tag2idx['<pad>'], tag2idx['<start>'], tag2idx['<end>'], tag2idx['entailed'], tag2idx['not_entailed'] = 0, 1, 2, 3, 4
    
    #### for scibert only 
    
    # tag2idx["PREAMBLE"] = 3 
    # tag2idx["ANALYSIS"] = 4
    # tag2idx["NONE"] = 5
    # tag2idx["FAC"] = 6
    # tag2idx["RLC"] = 7
    # tag2idx["ARG_PETITIONER"] = 8
    # tag2idx["STA"] = 9
    # tag2idx["PRE_RELIED"] = 10
    # tag2idx["RPC"] = 11
    # tag2idx["ISSUE"] = 12
    # tag2idx["RATIO"] = 13
    # tag2idx["PRE_NOT_RELIED"] = 14
    # tag2idx["ARG_RESPONDENT"] = 15

    # iterate over documents

    for doc in idx_order:
        start_index_curr = 0
        doc_x, doc_y = [], []
        word_doc_x, word_doc_y = [], []

        sent_per_doc = []
        with open(args.data_path + doc +".txt") as fp:

            # iterate over sentences
            for sent in fp:
                # print(sent)
                try:
                    sent_x, sent_y = sent.strip().split('\t')

                except ValueError:
                    continue

                # cleanse text, map words and tags
                if not args.pretrained:
                    
                    tokens = tokenizer.tokenize(sent_x)
                    
                    y_negative_labels, x_negative_sentence = \
                    add_negative_examples_and_change_dataset(sent_x, sent_y, tag2idx)

                    label_description_tokens = tokenizer.tokenize(descriptions_dict[sent_y])
                    if len(tokens) >= 50:
                        tokens = tokens[0:50]
                    tokens = ['[CLS]'] + tokens + ['[SEP]'] + label_description_tokens + ['[SEP]']
                    sent_x = tokenizer.convert_tokens_to_ids(tokens)

                else:
                    sent_x = list(map(float, sent_x.strip().split()[:args.emb_dim]))
                # sent_y = tag2idx[sent_y.strip()]

                if sent_x != []:
                    correct_sentence_idx = random.randint(0,len(x_negative_sentence))
                    x_negative_sentence.insert(correct_sentence_idx, sent_x)
                    y_negative_labels.insert(correct_sentence_idx, tag2idx["not_entailed"])

                    for sen, lab in zip(x_negative_sentence, y_negative_labels):
                      word_doc_x.append(tokenizer.decode(sen))
                      word_doc_y.append(lab)
                    doc_x.extend(x_negative_sentence)
                    doc_y.extend(y_negative_labels)
                    # doc_x.append(sent_x)
                    # doc_y.append(sent_y)


        x.append(doc_x)
        y.append(doc_y)
     

        full_word_doc_x.append(word_doc_x)
        full_word_doc_y.append(word_doc_y)
        sent_per_doc = []
        label_per_doc = []

        sent_number = -1
        first_time = 0


    return x, y, word2idx, tag2idx, []


def prepare_legalbert_embeddings(idx_order, args):
    x, y, targets = [], [], []
    original_sentences = {}

    word2idx = defaultdict(lambda: len(word2idx))
    tag2idx = defaultdict(lambda: len(tag2idx))

    # map the special symbols first
    word2idx['<pad>'], word2idx['<unk>'], word2idx['[CLS]'], word2idx['[SEP]'] = 0, 1, 2, 3
    tag2idx['<pad>'], tag2idx['<start>'], tag2idx['<end>'] = 0, 1, 2

    # iterate over documents

    for doc in idx_order:
        start_index_curr = 0
        doc_x, doc_y = [], []

        sent_per_doc = []
        with open(args.data_path + doc +".txt") as fp:

            # iterate over sentences
            for sent in fp:
                # print(sent)
                try:
                    sent_x, sent_y = sent.strip().split('\t')
                except ValueError:
                    continue

                # cleanse text, map words and tags
                if not args.pretrained:
                    tokens = tokenizer.tokenize(sent_x)
                    if len(tokens) >= 50:
                        tokens = tokens[0:50]
                    tokens = ['[CLS]'] + tokens + ['[SEP]']
                    sent_x = tokenizer.convert_tokens_to_ids(tokens)
                else:
                    sent_x = list(map(float, sent_x.strip().split()[:args.emb_dim]))
                sent_y = tag2idx[sent_y.strip()]

                if sent_x != []:
                    doc_x.append(sent_x)
                    doc_y.append(sent_y)


        x.append(doc_x)
        y.append(doc_y)


    return x, y, word2idx, tag2idx, original_sentences




'''
    This file prepares the numericalized data in the form of lists, to be used in training mode.
    idx_order is the order of documents in the dataset.

        x:  list[num_docs, sentences_per_doc, words_per_sentence]       if pretrained = False
            list[num_docs, sentences_per_doc, sentence_embedding_dim]   if pretrained = True
        y:  list[num_docs, sentences_per_doc]
'''
def prepare_data(idx_order, args):
    x, y = [], []
    original_sentences = {}

    word2idx = defaultdict(lambda: len(word2idx))
    tag2idx = defaultdict(lambda: len(tag2idx))

    # map the special symbols first
    word2idx['<pad>'], word2idx['<unk>'] = 0, 1
    tag2idx['<pad>'], tag2idx['<start>'], tag2idx['<end>'] = 0, 1, 2
    # iterate over documents

    for doc in idx_order:
        doc_x, doc_y = [], [] 
        start_index_curr = 0
        with open(args.data_path + doc + '.txt') as fp:
            # iterate over sentences
            for sent in fp:
                try:
                	sent_x, sent_y = sent.strip().split('\t')
                except Exception as e:
                    continue
                if not args.pretrained:
                    sent_x = sent_x.strip().lower().translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
                    sent_x = list(map(lambda x: word2idx[x], sent_x.split()))
                else:
                    sent_x = list(map(float, sent_x.strip().split()[:args.emb_dim]))
                sent_y = tag2idx[sent_y.strip()]
                if sent_x != []:
                    doc_x.append(sent_x)
                    doc_y.append(sent_y)
        
        x.append(doc_x)
        y.append(doc_y)


        sent_per_doc = []
        label_per_doc = []

        sent_number = -1
        first_time = 0

        # with open(args.data_docs_original + doc + '.txt') as fp:
        #     # iterate over sentences
            
        #     for sent in fp:
        #         sent_number+= 1
        #         try:
        #         	sent_x, sent_y = sent.strip().split('\t')
        #         except Exception as e:
        #         	continue
        #         if "the judgment of the court was delivered by " in sent_x.lower() and first_time==0:
        #           start_index_curr = sent_number
        #           first_time = 1
                  
        #         if sent_x != []:
        #             sent_per_doc.append(sent_x)
        #             label_per_doc.append(sent_y)
        

        # if start_index_curr != 0:

        #   sent_per_doc = sent_per_doc[start_index_curr+1:]
        #   label_per_doc = label_per_doc[start_index_curr+1:]

        #   x[-1] = x[-1][start_index_curr+1:]
        #   y[-1] = y[-1][start_index_curr+1:]

        # original_sentences[doc] = sent_per_doc.copy()

    return x, y, word2idx, tag2idx, original_sentences


'''
    This file prepares the numericalized data in the form of lists, to be used in inference mode.
    idx_order is the order of documents in the dataset.

        x:  list[num_docs, sentences_per_doc, words_per_sentence]       if pretrained = False
            list[num_docs, sentences_per_doc, sentence_embedding_dim]   if pretrained = True
'''
def prepare_data_inference(idx_order, args, sent2vec_model):
    x = []

    # iterate over documents
    for doc in idx_order:
        doc_x = []

        with open(args.data_path + doc + '.txt') as fp:
            
            # iterate over sentences
            for sent in fp:
                sent_x = sent.strip()

                # cleanse text, map words and tags
                if not args.pretrained:
                    sent_x = sent_x.lower().translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
                    sent_x = list(map(lambda x: args.word2idx[x] if x in args.word2idx else args.word2idx['<unk>'], sent_x.split()))
                else:
                    sent_x = sent2vec_model.embed_sentence(sent_x).flatten().tolist()[:args.emb_dim]
                
                if sent_x != []:
                    doc_x.append(sent_x)
                    
        x.append(doc_x)

    return x
    
# import string
# from collections import defaultdict
# from transformers import AutoTokenizer, AutoModel
# tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')

# '''
#     This function constructs folds that have a balanced category distribution.
#     Folds are stacked up together to give the order of docs in the main data.
    
#     idx_order defines the order of documents in the data. Each sequence of (docs_per_fold) documents in idx_order can be treated as a single fold, containing documents balanced across each category.
# '''
# def prepare_folds(args):
#     with open(args.cat_path) as fp:
        
#         categories = []
#         for line in fp:
#             _, docs = line.strip().split('\t')
#             docs = docs.strip().split(' ')
#             categories.append(docs)

#     # categories: list[category, docs_per_category]

#     categories.sort(key = lambda x: len(x))
#     n_docs = len(sum(categories, []))
#     assert n_docs == args.dataset_size, "invalid category list"
           
#     docs_per_fold = args.dataset_size // args.num_folds   
#     folds = [[] for f in range(docs_per_fold)]
    
#     # folds: list[num_folds, docs_per_fold]
    
#     f = 0
#     for cat in categories:
#         for doc in cat:
#             folds[f].append(doc)
#             f = (f + 1) % 5

#     # list[num_folds, docs_per_fold] --> list[num_folds * docs_per_fold]
#     idx_order = sum(folds, [])
#     return idx_order

# '''
#     This file prepares the numericalized data in the form of lists, to be used in training mode.
#     idx_order is the order of documents in the dataset.

#         x:  list[num_docs, sentences_per_doc, words_per_sentence]       if pretrained = False
#             list[num_docs, sentences_per_doc, sentence_embedding_dim]   if pretrained = True
#         y:  list[num_docs, sentences_per_doc]
# '''
# def prepare_data(idx_order, args):
#     x, y = [], []

#     word2idx = defaultdict(lambda: len(word2idx))
#     tag2idx = defaultdict(lambda: len(tag2idx))

#     # map the special symbols first
#     word2idx['<pad>'], word2idx['<unk>'],word2idx['[CLS]'],word2idx['[SEP]'] = 0, 1,2,3
#     tag2idx['<pad>'], tag2idx['<start>'], tag2idx['<end>'] = 0, 1, 2

#     # iterate over documents
#     for doc in idx_order:
#         doc_x, doc_y = [], [] 

#         with open(args.data_path + doc + '.txt') as fp:
#             print(fp)
#             # iterate over sentences
#             for sent in fp:
#                 #print(sent)
#                 try:
#                 	sent_x, sent_y = sent.strip().split('\t')
#                 except ValueError:
#                 	continue

#                 # cleanse text, map words and tags
#                 if not args.pretrained:
#                     tokens=tokenizer.tokenize(sent_x)
#                     if len(tokens)>=50:
#                         tokens=tokens[0:50]
#                     tokens=['[CLS]']+tokens+['[SEP]']
#                     sent_x=tokenizer.convert_tokens_to_ids(tokens)
#                     #sent_x = sent_x.strip().lower().translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
#                     #sent_x="[CLS] "+sent_x+" [SEP]"
#                     #sent_x = list(map(lambda x: word2idx[x], sent_x.split()))
#                 else:
#                     sent_x = list(map(float, sent_x.strip().split()[:args.emb_dim]))
#                 sent_y = tag2idx[sent_y.strip()]

#                 if sent_x != []:
#                    # print(sent_x)
#                     doc_x.append(sent_x)
#                     doc_y.append(sent_y)
        
#         x.append(doc_x)
#         y.append(doc_y)

#     return x, y, word2idx, tag2idx

# '''
#     This file prepares the numericalized data in the form of lists, to be used in BERT inference mode.
#     idx_order is the order of documents in the dataset.

#         x:  list[num_docs, sentences_per_doc, words_per_sentence]       if pretrained = False
#             list[num_docs, sentences_per_doc, sentence_embedding_dim]   if pretrained = True
# '''
# def prepare_data_BERT_inference(idx_order, args, sent2vec_model):
#     x = []

#     # iterate over documents
#     for doc in idx_order:
#         doc_x = []

#         with open(args.data_path + doc + '.txt') as fp:
            
#             # iterate over sentences
#             for sent in fp:
#                 sent_x = sent.strip()

#                 # cleanse text, map words and tags
#                 if not args.pretrained:
#                     tokens=tokenizer.tokenize(sent_x)
#                     if len(tokens)>=50:
#                         tokens=tokens[0:50]
#                     tokens=['[CLS]']+tokens+['[SEP]']
#                     sent_x=tokenizer.convert_tokens_to_ids(tokens)
#                 else:
#                     sent_x = sent2vec_model.embed_sentence(sent_x).flatten().tolist()[:args.emb_dim]
                
#                 if sent_x != []:
#                     doc_x.append(sent_x)
#                 print(sent_x)
                    
#         x.append(doc_x)

#     return x
    


# '''
#     This file prepares the numericalized data in the form of lists, to be used in inference mode.
#     idx_order is the order of documents in the dataset.

#         x:  list[num_docs, sentences_per_doc, words_per_sentence]       if pretrained = False
#             list[num_docs, sentences_per_doc, sentence_embedding_dim]   if pretrained = True
# '''
# def prepare_data_inference(idx_order, args, sent2vec_model):
#     x = []

#     # iterate over documents
#     for doc in idx_order:
#         doc_x = []

#         with open(args.data_path + doc + '.txt') as fp:
            
#             # iterate over sentences
#             for sent in fp:
#                 sent_x = sent.strip()

#                 # cleanse text, map words and tags
#                 if not args.pretrained:
#                     sent_x = sent_x.lower().translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
#                     sent_x = list(map(lambda x: args.word2idx[x] if x in args.word2idx else args.word2idx['<unk>'], sent_x.split()))
#                 else:
#                     sent_x = sent2vec_model.embed_sentence(sent_x).flatten().tolist()[:args.emb_dim]
                
#                 if sent_x != []:
#                     doc_x.append(sent_x)
#                 print(sent_x)
                    
#         x.append(doc_x)

#     return x
    

