import torch
import time
import json
import os
import sys
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')

from logger_file import fetch_logger
logger = fetch_logger('train file logger')


'''
  Randomly shuffle the data and divide into batches
'''
def batchify(x, y, batch_size):
    idx = list(range(len(x)))
    random.shuffle(idx)
    
    # convert to numpy array for ease of indexing
    x = np.array(x)[idx]
    y = np.array(y)[idx]
    
    i = 0
    while i < len(x):
        j = min(i + batch_size, len(x))
        
        batch_idx = idx[i : j]
        batch_x = x[i : j]
        batch_y = y[i : j]
        
        yield batch_idx, batch_x, batch_y
        
        i = j


'''
    Perform a single training step by iterating over the entire training data once. Data is divided into batches.
'''
def train_step(model, opt, x, y, batch_size):
    ## x: list[num_examples, sents_per_example, features_per_sentence]
    ## y: list[num_examples, sents_per_example]
    
    model.train()
    
    total_loss = 0
    y_pred = [] # predictions
    y_gold = [] # gold standard
    idx = [] # example index
    for i, (batch_idx, batch_x, batch_y) in enumerate(batchify(x, y, batch_size)):
       
        pred, _ = model(batch_x, batch_y)
        
        loss = model._loss(batch_y)        

        opt.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        opt.step()
        
        total_loss += loss.item()
     
        y_pred.extend(pred)
        y_gold.extend(batch_y)
        idx.extend(batch_idx)
    # print(model.fc_attention.weight.grad)
    
    
        
    assert len(sum(y, [])) == len(sum(y_pred, [])), "Mismatch in predicted"
    
    return total_loss / (i + 1), idx, y_gold, y_pred


# descriptions_dict = {'FAC':'This is facts FAC',
#     'RLC' : 'This is Ruling of Lower Court RLC',
#     'ARG':'argues sought an Argument ARG',
#     'STA' : 'This is Stature STA',
#     'PRE' : 'Precedent or previous lower court ruling PRE',
#     'RATIO' : 'This is Ratio',
#     'RPC' : 'Conclusion and court judgement RPC'}


descriptions_dict = {'FAC':'This is facts FAC',
  'RLC' : 'This is Ruling of Lower Court RLC',
  'ARG':'This is an Argument ARG',
  'STA' : 'This is Stature STA',
  'PRE' : 'This is Precedent PRE',
  'Ratio' : 'This is Ratio',
  'RPC' : 'This is present court judgement RPC'}

def get_tokenised_label_descriptions():
  labels_to_token_ids = {}
  for label in descriptions_dict:
    token_to_ids = tokenizer.tokenize(descriptions_dict[label])
    labels_to_token_ids[label] = tokenizer.convert_tokens_to_ids(token_to_ids)
  return labels_to_token_ids

def modify_validation_dataset(input_data, labels, labels_to_token_ids):

  new_input_data = []
  new_labels = []

  for doc, labels in zip(input_data, labels):

    curr_doc_inputs, curr_doc_labels = [], []

    for doc_sents, sent_labels in zip(doc, labels):


      if sent_labels == 4: #original sentence
        
        original_sent_end_index = doc_sents.index(102)
        original_sent = doc_sents[:original_sent_end_index]
        original_label = tokenizer.decode(doc_sents[original_sent_end_index:]).split(' ')[-2].upper()
        negative_sentences_examples = []
        negative_sentences_labels = []

        for entailed_label in labels_to_token_ids:
          if entailed_label.upper()!=original_label.upper():
            new_sent = original_sent + [102] + labels_to_token_ids[entailed_label] + [102]
            negative_sentences_examples.append(new_sent)
            negative_sentences_labels.append(3)
        negative_sentences_examples = random.sample(negative_sentences_examples, random.randint(5,6))
        negative_sentences_labels = negative_sentences_labels[:len(negative_sentences_examples)]
        # random.shuffle(negative_sentences_examples)

        correct_sentence_idx = random.randint(0,len(negative_sentences_examples)-1)
        negative_sentences_examples.insert(correct_sentence_idx, doc_sents)
        negative_sentences_labels.insert(correct_sentence_idx, 4)

        curr_doc_inputs.extend(negative_sentences_examples)
        curr_doc_labels.extend(negative_sentences_labels)
    

    new_input_data.append(curr_doc_inputs)
    new_labels.append(curr_doc_labels)  

  # for sent, lab in zip(new_input_data[0][:200],new_labels[0][:200]):
  #   print(tokenizer.decode(sent)+'\t'+str(lab))


  # import sys
  # sys.exit(1)

  # len input data is 5 or 10, dpeending on your val size,
  # 
  return new_input_data, new_labels


'''
    Perform a single evaluation step by iterating over the entire training data once. Data is divided into batches.
'''
def val_step(model, x, y, batch_size):
    ## x: list[num_examples, sents_per_example, features_per_sentence]
    ## y: list[num_examples, sents_per_example]
    
    model.eval()
    all_scores = []
    all_inputs = []
    total_loss = 0
    y_pred = [] # predictions
    y_gold = [] # gold standard
    idx = [] # example index
    attention_weights = []
    with torch.no_grad():
      for i, (batch_idx, batch_x, batch_y) in enumerate(batchify(x, y, batch_size)):
          all_inputs.extend(batch_x)
          pred, scores = model(batch_x, batch_y)
          all_scores.append(scores)
          # attention_weights.extend(attention_per_head.permute(1,0,2,3,4))
          loss = model._loss(batch_y)
          total_loss += loss.item()
      
          y_pred.extend(pred)
          y_gold.extend(batch_y)
          idx.extend(batch_idx)
          
      assert len(sum(y, [])) == len(sum(y_pred, [])), "Mismatch in predicted"
    model.train()
    
    return total_loss / (i + 1), idx, y_gold, y_pred, all_scores, all_inputs

'''
    Infer predictions for un-annotated data
'''
def infer_step(model, x):
    ## x: list[num_examples, sents_per_example, features_per_sentence]
    
    model.eval()
    y_pred = model(x) # predictions
    
    return y_pred    


def pick_sentence_with_highest_probability(probs_tensor, start, end, doc_num, preds_sublist):


  probs_tensor = probs_tensor.squeeze(dim=0)
  probabilities_sub_tensor = probs_tensor[start:end] #prospective_sentences*n_tags
  pos = probabilities_sub_tensor.max(dim=1).values.argmax()

  if 4 in preds_sublist:

    idx_list = []
    
    for pred in preds_sublist:
      if pred == 3:
        idx_list.append([False]*5)
      else:
        idx_list.append([False]*3 + [True]*2)
    condition  = torch.tensor(idx_list)
    # print(condition)
 
    # idx_tensor = torch.tensor(idx_list).to('cuda')
    # probabilities_sub_tensor = torch.index_select(probabilities_sub_tensor, 0, idx_tensor)

    probabilities_sub_tensor = probabilities_sub_tensor.where(condition.to('cuda'), torch.tensor(-1000.0).to('cuda'))
    probabilities_sub_tensor = probabilities_sub_tensor.max(dim=1)
    pos = probabilities_sub_tensor.values.argmax() 

  return pos

def statistics(data_state, tag2idx):
    idx, gold, pred = data_state['idx'], data_state['gold'], data_state['pred']
    
    rev_tag2idx = {v: k for k, v in tag2idx.items()}
    tags = [rev_tag2idx[i] for i in range(len(tag2idx)) if rev_tag2idx[i] not in ['<start>', '<end>', '<pad>']]
    
    # flatten out
    gold = sum(gold, [])
    pred = sum(pred, [])
    
    print(classification_report(gold, pred, target_names = tags, digits = 3))

'''
    Report all metrics in format using sklearn.metrics.classification_report
'''
def statistics_for_entailment(data_state, tag2idx):
    idx, gold, pred = data_state['idx'], data_state['gold'], data_state['pred']
    
    rev_tag2idx = {v: k for k, v in tag2idx.items()}
    tags = [rev_tag2idx[i] for i in range(len(tag2idx)) if rev_tag2idx[i] not in ['<start>', '<end>', '<pad>']]
    all_doc_lines = []

    doc_set = data_state['input_data']
    probs = data_state['probs']

    actual_golds_all, actual_preds_all = [], []

    doc_num = 0
    # print(len(probs), probs[0].shape, len(doc_set[0]))
    for each_doc_labels, each_doc_preds, each_doc in zip(gold, pred, doc_set):
     
      actual_golds, actual_preds = [], []

      prev_line = str(each_doc[0]).split('102')[0]

      prev_label = each_doc_labels[0]

      prev_start_index = 0

      itr = 0


      if prev_label == 4:
        actual_golds.append(tokenizer.decode(each_doc[0]).split(' ')[-2].upper())
        all_doc_lines.append(str(tokenizer.decode(each_doc[0]))+" gold label : "+str(each_doc_labels[0])+ " pred label : "+str(each_doc_preds[0]) + " probabilities :"+str(probs[doc_num].squeeze(dim=0)[itr][:])+"\n")

        
      flag = 0
      for curr_line, curr_label, curr_pred in zip(each_doc, each_doc_labels, each_doc_preds):
        
        if flag == 0:
          flag = 1
        else:
        
          itr+=1
          try:
            all_doc_lines.append(str(tokenizer.decode(curr_line))+" gold label : "+str(curr_label)+ " pred label : "+str(curr_pred) + " probabilities :"+str(probs[doc_num].squeeze(dim=0)[itr][:])+"\n")
          except Exception as e:
            print(e)
            print(itr, probs[doc_num].shape, probs[doc_num].squeeze(dim=0).shape)


          if str(curr_line).split('102')[0] != prev_line or itr==len(each_doc)-1:
            # next set of neg/pos sentences 
            start = prev_start_index
            end = itr
            sentence_num = prev_start_index + pick_sentence_with_highest_probability(probs[doc_num], start, end, doc_num, each_doc_preds[start:end])

            actual_preds.append(tokenizer.decode(each_doc[sentence_num]).split(' ')[-2].upper())
            prev_start_index = itr
            prev_line = str(curr_line).split('102')[0]

          if curr_label == 4:
            actual_golds.append(tokenizer.decode(curr_line).split(' ')[-2].upper())

      actual_golds_all.extend(actual_golds[:len(actual_preds)])
      actual_preds_all.extend(actual_preds)
      

      print(len(actual_golds), len(actual_preds), sum([1 for lab in each_doc_labels if lab == 4]))


      with open('/content/drive/MyDrive/IIT_law_ai/semantic_segmentation/all_doc_lines_{}.txt'.format(doc_num), 'w') as fp:
        fp.writelines(all_doc_lines)


      doc_num+=1
   
    # flatten out
    gold = sum(gold, [])
    pred = sum(pred, [])
    

    gold_only_original, pred_only_original = [], []
    
    print(classification_report(actual_golds_all, actual_preds_all))
    print(classification_report(gold, pred))
    # , target_names = tags, digits = 3))
    # logger.info(classification_report(gold, pred, target_names = tags, digits = 3))



def create_comparison_dataset(idx_order, args, tag2idx, val_fold = 0):

  doc_x, doc_y = [], []
  final_res = []
  idx2tag = {v: k for k, v in tag2idx.items()}
  data_state_path = args.save_path + 'data_state' + str(val_fold) +'.json'
  f = open(data_state_path)

  data = json.load(f)
  sent_iterator_global = 0
  # print(data['gold'][0])
  # print(len((data['gold'][0])))
  # print("above was data gold")


  print(idx_order[120:])

  for idx in data['idx']:

    file_name = idx_order[120:][idx]
    print(file_name)
    with open('/content/drive/MyDrive/IIT_law_ai/semantic_segmentation/dataset/IN-train-set/' + file_name + '.txt') as fp:
      
      for sent in fp:
          try:
            sent_x, sent_y = sent.strip().split('\t')
          except ValueError:
            continue
          doc_x.append(sent_x)
          doc_y.append(sent_y)

      sent_iterator = 0
      while sent_iterator < len(doc_x):
        final_res.append([file_name, sent_iterator, doc_x[sent_iterator], doc_y[sent_iterator], tag2idx[data['pred'][0][sent_iterator_global]], tag2idx[data['gold'][0][sent_iterator_global]]])
        sent_iterator+=1
        sent_iterator_global += 1
      
      
    
    doc_x, doc_y = [], []

  import pandas as pd 
  
  df = pd.DataFrame(data = final_res, columns = ['filename','sent_num', 'sent', 'label_from_doc', 'label_pred', 'label_gold'])
  df.to_csv('fold_4_best_india_model_jurix_150_preds_comparison.csv')


'''
    Train the model on entire dataset and report loss and macro-F1 after each epoch.
'''

def learn(model, x, y, tag2idx, val_fold, args, idx_order = [], original_docs = {}):
    



    attention_df = []
    samples_per_fold = args.dataset_size // args.num_folds
    print(samples_per_fold, args.num_folds, args.dataset_size)
    

    if args.use_scibert_data == False:

      val_idx = list(range(val_fold * samples_per_fold, val_fold * samples_per_fold + samples_per_fold))
      train_idx = list(range(val_fold * samples_per_fold)) + list(range(val_fold * samples_per_fold + samples_per_fold, args.dataset_size))
      
      train_x = [x[i] for i in train_idx]
      train_y = [y[i] for i in train_idx]
      val_x = [x[i] for i in val_idx]
      val_y = [y[i] for i in val_idx]
      print(len(train_x), train_idx)
    
    else:

      print("here and hence wrong")

      train_idx = list(range(0, 184))
      val_idx = list(range(184, 214))
          
      train_x = [x[i] for i in train_idx]
      train_y = [y[i] for i in train_idx]
      val_x = [x[i] for i in val_idx]
      val_y = [y[i] for i in val_idx]


    val_idx_org = val_idx
   


    if args.use_attention:

      opt = torch.optim.Adam(
        [
                  {'params': model.gru.parameters(), 'lr': 1e-2},
                  {'params': model.dropout.parameters(), 'lr': 1e-4},
                  {'params': model.fc.parameters(), 'lr': 1e-4},
                  {'params': model.crf.parameters(), 'lr': 1e-2},
        ],
        lr = args.lr, weight_decay = args.reg)

  

    else:
      print("normal opt params")
      opt = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.reg)

    

    print("{0:>7}  {1:>10}  {2:>6}  {3:>10}  {4:>6}".format('EPOCH', 'Tr_LOSS', 'Tr_F1', 'Val_LOSS', 'Val_F1'))
    print("-----------------------------------------------------------")
    logger.info("{0:>7}  {1:>10}  {2:>6}  {3:>10}  {4:>6}".format('EPOCH', 'Tr_LOSS', 'Tr_F1', 'Val_LOSS', 'Val_F1'))
    logger.info("-----------------------------------------------------------")
    
    best_val_f1 = 0.0
    
    model_state = {}
    data_state = {}
    
    start_time = time.time()
    best_attention_weights = ()

    idx2tag = {}
    attention_weights = []
    labels_to_token_ids = get_tokenised_label_descriptions()

    for tag in tag2idx:
      idx2tag[tag2idx[tag]] = tag

    
    # val_x, val_y = modify_validation_dataset(val_x, val_y, labels_to_token_ids)

    for epoch in range(1, args.epochs + 1):
        


        # train_loss, train_idx, train_gold, train_pred = train_step(model, opt, train_x, train_y, args.batch_size)
        start = time.time()
        print("going for val")
        val_loss, val_idx, val_gold, val_pred, scores, inputs = val_step(model, val_x, val_y, args.batch_size)
        print(time.time()- start)
        print("this is the time taken")

        train_f1 = f1_score(sum(train_gold, []), sum(train_pred, []), average = 'macro')
        val_f1 = f1_score(sum(val_gold, []), sum(val_pred, []), average = 'macro')
        


        ### added lr decay

        # lr_decay.step(val_loss)

        

        if epoch % args.print_every == 0:
            print("{0:7d}  {1:10.3f}  {2:6.3f}  {3:10.3f}  {4:6.3f}".format(epoch, train_loss, train_f1, val_loss, val_f1))
            logger.info("{0:7d}  {1:10.3f}  {2:6.3f}  {3:10.3f}  {4:6.3f}".format(epoch, train_loss, train_f1, val_loss, val_f1))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model_state = {'epoch': epoch, 'arch': model, 'name': model.__class__.__name__, 'state_dict': model.state_dict(), 'best_f1': val_f1, 'optimizer' : opt.state_dict()}
            data_state = {'idx': val_idx, 'loss': val_loss, 'gold': val_gold, 'pred': val_pred, 'input_data':inputs, 'probs':scores}
            best_attention_weights = (attention_weights, val_gold, val_idx, val_pred)

    # attention_weights,  val_gold, val_indices, val_pred = best_attention_weights


    # for doc_id in range(len(val_gold)):
    
    #   true_label_list = val_gold[doc_id]
    #   pred_label_list = val_pred[doc_id]
    #   doc_len = len(val_gold[doc_id])



      


    #   attention_head1 = attention_weights[doc_id][0][0]
    #   attention_head2 = attention_weights[doc_id][0][1]
    #   attention_head3 = attention_weights[doc_id][0][2]
    #   attention_head4 = attention_weights[doc_id][0][3]


      # for itr, sentence in enumerate(original_docs[idx_order[val_idx_org[0]+val_indices[doc_id]]]):
        
      #   curr_doc_df = []
      #   curr_doc_df.append(idx_order[val_idx_org[0]+val_indices[doc_id]])
        
      #   ### add sentence
      #   curr_doc_df.append(itr)
      #   curr_doc_df.append(sentence)

      #   ### true labels
      #   curr_doc_df.append(idx2tag[true_label_list[itr]])
      #   #### pred labels
      #   curr_doc_df.append(idx2tag[pred_label_list[itr]])
      #   ### add top 5 attention scores for each head
      #   top_5_attentions_head = []
      #   top_5_attentions_head.append(sorted(attention_head1[itr].argsort()[-5:].tolist(), reverse = False))
      #   top_5_attentions_head.append(sorted(attention_head2[itr].argsort()[-5:].tolist(), reverse = False))
      #   top_5_attentions_head.append(sorted(attention_head3[itr].argsort()[-5:].tolist(), reverse = False))
      #   top_5_attentions_head.append(sorted(attention_head4[itr].argsort()[-5:].tolist(), reverse = False))
        
      #   for top_indices_per_head in top_5_attentions_head:
      #     curr_doc_df.append([original_docs[idx_order[val_idx_org[0]+val_indices[doc_id]]][sent_index] for sent_index in top_indices_per_head])
      #     curr_doc_df.append([top_indices_per_head])
      #     curr_doc_df.append([idx2tag[true_label_list[sent_index]] for sent_index in top_indices_per_head])

      #   # for item in curr_doc_df:
      #   #   print(item)

      #   # print("sentence : "+str(sentence))
      #   # print("label : "+str(label_list[itr]))
      #   attention_df.append(curr_doc_df)


    
      # for head in range(4):
      #   fig, ax = plt.subplots(1,1)
      #   x_ticks = np.arange(0 ,50)
      #   y_ticks = x_ticks.copy()[::-1]
      #   x_ticks_labels = label_list[-50:]
      #   doc_len = len(val_gold[doc_id])
      #   plt.imshow(attention_weights[doc_id][0][head][len(val_gold[doc_id])-50:len(val_gold[doc_id]), len(val_gold[doc_id])-50 :len(val_gold[doc_id])].cpu().detach().numpy())
      #   plt.colorbar()

      #   fig.set_size_inches(20, 20)

      #   ax.set_xticks(x_ticks)
      #   ax.set_yticks(y_ticks)
      #   ax.set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=18)
      #   ax.set_yticklabels(x_ticks_labels, rotation='horizontal', fontsize=18)
        
        # attention_viz_path = '/content/drive/MyDrive/IIT_law_ai/semantic_segmentation/attention_viz_lr_0.01/'+str(idx_order[val_idx_org[0]+val_indices[doc_id]])+"/"
        # if not os.path.exists(os.path.dirname(attention_viz_path)):
        #   try:
        #       os.makedirs(os.path.dirname(attention_viz_path))
        #   except OSError as exc: # Guard against race condition
        #       if exc.errno != errno.EEXIST:
        #           raise
        
        # plt.savefig('/content/drive/MyDrive/IIT_law_ai/semantic_segmentation/attention_viz_lr_0.01/'+str(idx_order[val_idx_org[0]+val_indices[doc_id]])+'/attention_head_'+str(head)+'_doc_'+str(idx_order[val_idx_org[0]+val_indices[doc_id]])+'_epochs_'+str(args.epochs)+'.png')
        
    end_time = time.time()
    print("Dumping model and data ...", end = ' ')


    torch.save(model_state, args.save_path + 'model_state' + str(val_fold) + '.tar')

    with open(args.save_path + 'data_state' + str(val_fold) + '.json', 'w') as fp:
        json.dump(data_state, fp)
    create_comparison_dataset(idx_order, args, idx2tag, val_fold)
    
    print("Done")    

    print('Time taken:', int(end_time - start_time), 'secs')

    statistics(data_state, tag2idx)


    return best_val_f1 , attention_df






