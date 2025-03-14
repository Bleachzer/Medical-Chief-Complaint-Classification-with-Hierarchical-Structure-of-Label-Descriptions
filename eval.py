
#%%
from config import *

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm, trange
import os
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from classifiers import CLASSIFIER_CLASSES
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from multiprocessing import Pool, cpu_count
from utils import *
from convert_example_to_features import *
from visualization import * 
from sklearn.metrics import precision_score, recall_score, f1_score
from eval_reports_analysis_utils import * 

# FIXME: save this in config 

do_reports_analysis = True
if do_visualization:
    vis = Visualizations(visdom_env_name)

    
import logging
logging.basicConfig(level=logging.INFO)


classifier_class = CLASSIFIER_CLASSES[classifier_type]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import random
import numpy as np
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

set_seed(42) # For reproductivity

#%%

def evaluate(eval_task_name, model, tokenizer):
    """
    eval_task_name: task name without file type

    return:
        accuracy: question accuracy

        preds: logits for all QC pair

        out_label_ids: label for all QC pair 0, 1
    """
    processor = MultiClassificationProcessor()
    eval_examples = processor.get_dev_examples(data_dir, f'{eval_task_name}.csv')
    label_list = processor.get_labels()
    eval_features = convert_examples_to_features(eval_examples, 
                                                    label_list, 
                                                    max_seq_length, 
                                                    tokenizer, 
                                                    output_mode)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

    #%%
    # FIXME: is there a need to use multi gpu here
    # if n_gpus > 1:
    #     model = torch.nn.DataParallel(model)
    # let's just use one for now
    eval_batch_size = per_gpu_eval_size

    eval_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)


    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Total batch size = %d", eval_batch_size)
    #%%

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2],  # change this if using xlnet
                        'labels':         batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            # reshaped_logits = logits.view(-1, 5) # 5: num of choices 
            # _, labels = torch.max(batch[3].view(-1, 5), 1)
            # loss_fct = CrossEntropyLoss()
            # tmp_eval_loss = loss_fct(reshaped_logits, labels)


            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = batch[3].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, batch[3].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    preds_list = np.argmax(preds, 1)
    labels_list = out_label_ids
    print("preds_list: ",preds_list)
    print("labels_list",labels_list)

    #%%
    eval_accuracy = sum([pred == label for pred, label in zip(preds_list, labels_list)]) / len(preds_list)
    microPrecision = precision_score(labels_list,preds_list,labels = list(range(num_labels)),average='micro')
    microRecall = recall_score(labels_list,preds_list,labels = list(range(num_labels)),average='micro')
    microF1score = f1_score(labels_list,preds_list,labels = list(range(num_labels)),average='micro')

    macroPrecision = precision_score(labels_list,preds_list,labels = list(range(num_labels)),average='macro')
    macroRecall = recall_score(labels_list,preds_list,labels = list(range(num_labels)),average='macro')
    macroF1score = f1_score(labels_list,preds_list,labels = list(range(num_labels)),average='macro')

    divPrecision = precision_score(labels_list,preds_list,labels = list(range(num_labels)),average = None)
    divRecall = recall_score(labels_list,preds_list,labels = list(range(num_labels)),average= None)
    divF1score = f1_score(labels_list,preds_list,labels = list(range(num_labels)), average = None)
    print('-------------##########################-------------')
    results = {
        'file': eval_task_name,
        'eval_loss': eval_loss,
        'eval_accuracy': eval_accuracy,
        'microPrecision': microPrecision,
        'microRecall': microRecall,
        'microF1score': microF1score,
        'macroPrecision': macroPrecision,
        'macroRecall': macroRecall,
        'macroF1score': macroF1score,
        'divPrecision': divPrecision,
        'divRecall': divRecall,
        'divF1score': divF1score,
        'param_stats': stats,
        'preds': preds_list,
        'labels': labels_list
        }
    return results, preds, out_label_ids
    
    
    


def evaluate_iterator(test_file_list, epoch_list):
    assert test_file_list != [] and epoch_list != []
    for epoch in epoch_list:
        cache_path = f'{cache_dir}epoch_{epoch}/'
        tokenizer = BertTokenizer.from_pretrained(cache_path, do_lower_case=True)
        # output only one logit
        model = classifier_class.from_pretrained(cache_path, num_labels=num_labels)

        model.to(device)
        for file in test_file_list:
            results, preds_raw, labels_raw = evaluate(file, model, tokenizer)
            results['epoch'] = epoch
            print(results)

            # write acc to a txt file 
            write_results(results)

            # draw accuracy plot
            accuracy_plot_helper(results['eval_accuracy'], epoch, file)
            
            # visualization: visdom localhost 8097
            if do_visualization:
                visualize_helper(results['eval_accuracy'], epoch, file)
            
            # get reports analysis: qtype-iscorrect figure, output file
            # if do_reports_analysis:
            #     get_reports_analysis(epoch, file,
            #                         preds_raw, labels_raw, 
            #                         data_dir=data_dir)


def write_results(results):
    output_eval_file = os.path.join(output_dir, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
            writer.write("%s = %s\n" % (key, str(results[key])))
        writer.write('\n\n')

# visualization
def visualize_helper(acc, epoch, file):
    if 'trainAll4000' == file:
        vis.plot_accuracy1(acc, epoch)
    if 'testAll4000' == file:
        vis.plot_accuracy2(acc, epoch)

def accuracy_plot_helper(acc, epoch, file):
    if 'train_reduce' == file:
        trainMedicine.append(acc)
    if 'test_reduce' == file:
        testMedicine.append(acc)

#%%
# trainMedicine = []
testMedicine = []

evaluate_iterator(eval_files, eval_epochs) # set in config.py
# plot and save acc_epoch line figure
acc_plot(testMedicine, eval_epochs, 'test')
