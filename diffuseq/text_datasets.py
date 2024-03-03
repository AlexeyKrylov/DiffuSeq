import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset

import torch
import json
import psutil
import datasets
from datasets import Dataset as Dataset2

def load_data_text(
    batch_size, 
    seq_len, 
    deterministic=False, 
    data_args=None, 
    split='train',
    loaded_vocab=None,
    loop=True,
    nofb=None,
    nofs=None
):
    """
    For a dataset, create a generator over (seqs, kwargs) pairs.

    Each seq is an (bsz, len, h) float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for some meta information.

    :param batch_size: the batch size of each returned pair.
    :param seq_len: the max sequence length (one-side).
    :param deterministic: if True, yield results in a deterministic order.
    :param data_args: including dataset directory, num of dataset, basic settings, etc.
    :param model_emb: loaded word embeddings.
    :param loaded_vocab: loaded word vocabs.
    :param loop: loop to get batch data or not.
    """

    training_data = get_corpus(data_args, seq_len, split=split, loaded_vocab=loaded_vocab, nofs=nofs)

    dataset = TextDataset(
        training_data,
        data_args
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=data_args.drop_last,
        shuffle=not deterministic,
        num_workers=0,
    )
    if loop:
        return infinite_loader(data_loader)

    else:
        if nofb is None:
            return iter(data_loader)

        res_iter = list()
        iter_dataloader = iter(data_loader)
        while nofb > 0:
            nofb -= 1
            res_iter.append(next(iter_dataloader))
        return iter(res_iter)

def infinite_loader(data_loader):
    while True:
        yield from data_loader

def helper_tokenize(data_args, sentence_lst, vocab_dict, seq_len):
    raw_datasets = Dataset2.from_dict(sentence_lst)

    def tokenize_function(examples):
        input_id_x = vocab_dict.encode_token(examples['src'])
        input_id_y = vocab_dict.encode_token(examples['trg'])
        result_dict = {'input_id_x': input_id_x, 'input_id_y': input_id_y}

        return result_dict

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=['src', 'trg'],
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    def merge_and_mask(group_lst):
        lst = []
        mask = []
        for i in range(len(group_lst['input_id_x'])):
            end_token = group_lst['input_id_x'][i][-1]
            src = group_lst['input_id_x'][i][:-1]
            trg = group_lst['input_id_y'][i][:-1]
            while len(src) + len(trg) > seq_len - 3:
                if len(src)>len(trg):
                    src.pop()
                elif len(src)<len(trg):
                    trg.pop()
                else:
                    src.pop()
                    trg.pop()
            src.append(end_token)
            trg.append(end_token)

            lst.append(src + [vocab_dict.sep_token_id] + trg)
            mask.append([0]*(len(src)+1))
        group_lst['input_ids'] = lst
        group_lst['input_mask'] = mask

        if data_args.debug_mode:
            print('-'*120)
            print("Max length of sequence in each thousand of examples: ", max([len(i) for i in lst]))
            print("Decentiles length of sequence in each thousand of examples: ", sorted([len(i) for i in lst])[len(lst) // 10::len(lst) // 10])
            print('-'*120)

        return group_lst
    
    tokenized_datasets = tokenized_datasets.map(
        merge_and_mask,
        batched=True,
        num_proc=1,
        desc=f"merge and mask",
    )
    
    def pad_function(group_lst):
        max_length = seq_len
        group_lst['input_ids'], pad_mask = _collate_batch_helper(group_lst['input_ids'], vocab_dict.pad_token_id, max_length, return_mask=True)
        group_lst['input_mask'] = _collate_batch_helper(group_lst['input_mask'], 1, max_length)
        group_lst["pad_mask"] = pad_mask
        return group_lst

    lm_datasets = tokenized_datasets.map(
        pad_function,
        batched=True,
        num_proc=1,
        desc=f"padding",
    )

    if data_args.debug_mode:
        print(lm_datasets, 'padded dataset')
        print('### tokenized_datasets...example X', lm_datasets['input_id_x'][0])
        print('### tokenized_datasets...example Y', lm_datasets['input_id_y'][0])
        print('### tokenized_datasets...example ids', lm_datasets['input_ids'][0])
        print('### tokenized_datasets...example mask', lm_datasets['input_mask'][0])

    raw_datasets = datasets.DatasetDict()
    raw_datasets['train'] = lm_datasets
    return raw_datasets


def get_corpus(data_args, seq_len, split='train', loaded_vocab=None, nofs=None):

    if nofs is None:
        nofs = 999999

    if data_args.debug_mode:
        print('#'*30, '\nLoading dataset')

    sentence_lst = {'src':[], 'trg': []}
    
    if split == 'train':
        if data_args.debug_mode:
            print('### Loading form the TRAIN set...')
        path = f'{data_args.data_dir}/train.jsonl'
    elif split == 'valid':
        if data_args.debug_mode:
            print('### Loading form the VALID set...')
        path = f'{data_args.data_dir}/valid.jsonl'
    elif split == 'test':
        if data_args.debug_mode:
            print('### Loading form the TEST set...')
        path = f'{data_args.data_dir}/test.jsonl'
    else:
        assert False, "invalid split for dataset"

    with open(path, 'r') as f_reader:
        for row in f_reader:
            sentence_lst['src'].append(json.loads(row)['src'].strip().replace("<page_title>", "<PAGESTART>").replace("</page_title>", "<PAGEEND>") \
                                    .replace("<section_title>", "<SECTIONSTART>").replace("</section_title>", "<SECTIONEND>") \
                                    .replace("<table>", "<TABLESTART>").replace("</table>", "<TABLEEND>") \
                                    .replace("<cell>", "<CELLSTART>").replace("</cell>", "<CELLEND>") \
                                    .replace("<col_header>", "<COLHEADERSTART>").replace("</col_header>", "<COLHEADEREND>") \
                                    .replace("<row_header>", "<ROWHEADERSTART>").replace("</row_header>", "<ROWHEADEREND>"))
            sentence_lst['trg'].append(json.loads(row)['trg'].strip())
            nofs -= 1
            if nofs <= 0:
                break

    if data_args.debug_mode:
        print('### Data samples...\n', sentence_lst['src'][:2], sentence_lst['trg'][:2])
        
    vocab_dict = loaded_vocab

    train_dataset = helper_tokenize(data_args, sentence_lst, vocab_dict, seq_len)
    return train_dataset


class TextDataset(Dataset):
    def __init__(self, text_datasets, data_args):
        super().__init__()
        self.text_datasets = text_datasets
        self.length = len(self.text_datasets['train'])
        self.data_args = data_args

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with torch.no_grad():
            out_kwargs = {}
            out_kwargs['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])
            out_kwargs['input_mask'] = np.array(self.text_datasets['train'][idx]['input_mask'])
            out_kwargs['attention_mask'] = np.array(self.text_datasets['train'][idx]['pad_mask'])

            return out_kwargs


def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False):
    result = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    mask_ = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result