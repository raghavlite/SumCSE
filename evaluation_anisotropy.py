import sys
import io, os
import numpy as np
import logging
import argparse
from prettytable import PrettyTable
import torch
import transformers
from transformers import AutoModel, AutoTokenizer
import tqdm
from functools import partialmethod
import time
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# Set PATHs
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval



def cal_avg_cosine(s1, s2):
    cos = torch.nn.CosineSimilarity(dim=-1)
    s1 = torch.tensor(s1).cuda()
    s2 = torch.tensor(s2).cuda()
    kk = []
    # import ipdb; ipdb.set_trace()
    # pbar = tqdm.tqdm(total=n)
    with torch.no_grad():
        for i in range(len(s1)):
            kk.append(cos(s1[i:i+1], s2).mean().item())
            # pbar.set_postfix({'cosine': sum(kk)/len(kk)})
            # pbar.update(1)
    return sum(kk) /len(kk)



def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str,
                        help="Transformers' model name or path")
    parser.add_argument("--pooler", type=str,
                        choices=['cls', 'cls_before_pooler', 'avg', 'avg_top2', 'avg_first_last'],
                        default='cls',
                        help="Which pooler to use")


    args = parser.parse_args()

    # Load transformers' model checkpoint
    model = AutoModel.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    def batcher(params, batch, max_length=None):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        sentences = [' '.join(s) for s in batch]

        # Tokenization
        if max_length is not None:
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
                max_length=max_length,
                truncation=True
            )
        else:
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
            )

        # Move to the correct device
        for k in batch:
            batch[k] = batch[k].to(device)

        # Get raw embeddings
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True, return_dict=True)
            last_hidden = outputs.last_hidden_state
            pooler_output = outputs.pooler_output
            hidden_states = outputs.hidden_states

        # Apply different poolers
        if args.pooler == 'cls':
            # There is a linear+activation layer after CLS representation
            return pooler_output.cpu()
        elif args.pooler == 'cls_before_pooler':
            return last_hidden[:, 0].cpu()
        elif args.pooler == "avg":
            return ((last_hidden * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(
                -1).unsqueeze(-1)).cpu()
        elif args.pooler == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch[
                'attention_mask'].sum(-1).unsqueeze(-1)
            return pooled_result.cpu()
        elif args.pooler == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(1) / \
                            batch['attention_mask'].sum(-1).unsqueeze(-1)
            return pooled_result.cpu()
        else:
            raise NotImplementedError


    start_time=time.time()
    # code to calculate anisotropy
    with open('../DATA/wiki1m_for_simcse.txt') as f:
        lines = f.readlines()[:100000]
    batch, embeds = [], []
    print('Get Sentence Embeddings....')
    for line in lines:
        batch.append(line.replace('\n', '').lower().split()[:32])
        if len(batch) >= 128:
            embeds.append(batcher(None, batch).detach().numpy())
            batch = []
    embeds.append(batcher(None, batch).detach().numpy())
    print('Calculate anisotropy....')
    embeds = np.concatenate(embeds, axis=0)
    cosineall = cal_avg_cosine(embeds, embeds)
    # print('Avg. Cos:', cosine)


    # code to calculate length anisotrophy
    with open('../DATA/wiki1m_for_simcse.txt') as f:
        lines = f.readlines()[:100000]
    batchlt, embedslt = [], []
    batchgt, embedsgt = [], []
    print('Get Sentence Embeddings....')
    for line in lines:
        if(len(line.split()) <= 10):
            batchlt.append(line.replace('\n', '').split())
            if len(batchlt) >= 128:
                embedslt.append(batcher(None, batchlt).detach().numpy())
                batchlt = []
        else:
            batchgt.append(line.replace('\n', '').split()[:32])
            if len(batchgt) >= 128:
                embedsgt.append(batcher(None, batchgt).detach().numpy())
                batchgt = []

    # print(np.histogram(batch, 25))
    embedslt.append(batcher(None, batchlt).detach().numpy())
    embedsgt.append(batcher(None, batchgt).detach().numpy())

    embedslt = np.concatenate(embedslt, axis=0)
    embedsgt = np.concatenate(embedsgt, axis=0)

    cosineltgt = cal_avg_cosine(embedslt, embedsgt)
    cosineltlt = cal_avg_cosine(embedslt, embedslt)
    cosinegtgt = cal_avg_cosine(embedsgt, embedsgt)
    print(f"No. <=10: {len(batchlt)}; No. >10: {len(batchgt)}")
    print("Avg. Cos (all, all)", "Avg. Cos (s,l):", "Avg. Cos (s,s):", "Avg. Cos (l, l):")
    print(cosineall, cosineltgt, cosineltlt, cosinegtgt)
    end_time=time.time()
    print(end_time-start_time)

if __name__ == "__main__":
    main()