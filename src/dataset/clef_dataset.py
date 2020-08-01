import random
import pickle
import logging
from tqdm.auto import tqdm
from torch.utils.data import IterableDataset, Dataset, DataLoader

logger = logging.getLogger()

class Retrieval_Trainset(Dataset):
    def __init__(self, id2query, id2doc, rel, split, num_neg, neg_value):
        
        self.rel = rel
        self.split = split
        self.id2doc = id2doc
        self.num_neg = num_neg
        self.neg_val = neg_value
        self.id2query = id2query
        
        # query ids in training split
        self.query_ids = set(self.split["train"])

        # for training
        self.positive_qd_pairs = []
        for query_id, value in tqdm(self.rel.items()):
            if query_id in self.query_ids and len(value["n"]) >= self.num_neg:
                for doc_id in value["p"]:
                    self.positive_qd_pairs.append((query_id, doc_id))

        logger.info("Number of positive query-document pairs in [train] set: {}".format(len(self.positive_qd_pairs)))

    def __len__(self):
        return len(self.positive_qd_pairs)

    def __getitem__(self, idx):
        
        qids, dids, queries, documents, y  = [], [], [], [], []
        
        # positive qd pair
        tmp = self.positive_qd_pairs[idx]
        
        if isinstance(tmp, tuple):
            # one positive qd pair
            (q, pos_d) = tmp
            qids.append(q)
            dids.append(pos_d)
            queries.append(self.id2query[q])
            documents.append(self.id2doc[pos_d])
            y.append(1)

            #negative qd pair
            neg_ds = random.sample(self.rel[q]["n"], self.num_neg)
            for neg_d in neg_ds:
                qids.append(q)
                dids.append(neg_d)
                queries.append(self.id2query[q])
                documents.append(self.id2doc[neg_d])
                y.append(self.neg_val)
        
        elif isinstance(tmp, list):
            # multiple positive qd pairs
            for (q, pos_d) in tmp:
                qids.append(q)
                dids.append(pos_d)
                queries.append(self.id2query[q])
                documents.append(self.id2doc[pos_d])
                y.append(1)

                #negative qd pair
                neg_ds = random.sample(self.rel[q]["n"], self.num_neg)
                for neg_d in neg_ds:
                    qids.append(q)
                    dids.append(neg_d)
                    queries.append(self.id2query[q])
                    documents.append(self.id2doc[neg_d])
                    y.append(self.neg_val)
        
        else:
            print(type(tmp))
            assert False
        
        return qids, dids, queries, documents, y
    
class Retrieval_Testset(Dataset):
    def __init__(self, id2query, id2doc, rel, split, mode, neg_value):
        
        self.rel = rel
        self.mode = mode
        self.split = split
        self.id2doc = id2doc
        self.neg_val = neg_value
        self.id2query = id2query
        
        # query ids in the split
        self.query_ids = set(self.split[self.mode])

        # for testing
        self.all_qd_pairs = []
        for query_id, value in tqdm(self.rel.items()):
            if query_id in self.query_ids:
                for doc_id in value["p"]:
                    self.all_qd_pairs.append([query_id, doc_id, 1])
                for doc_id in value["n"]:
                    self.all_qd_pairs.append([query_id, doc_id, self.neg_val])

        logger.info("Number of labelled query-document pairs in [{}] set: {}".format(self.mode, len(self.all_qd_pairs)))

    def __len__(self):
        return len(self.all_qd_pairs)

    def __getitem__(self, idx):
        
        (q, d, y) = self.all_qd_pairs[idx]
        query = self.id2query[q]
        document = self.id2doc[d]
        
        return q, d, query, document, y

def train_collate(batch):
    
    qids, dids, queries, documents, y = [], [], [], [], []
    for (b_qids, b_dids, b_queries, b_documents, b_y) in batch:
        qids.extend(b_qids)
        dids.extend(b_dids)
        queries.extend(b_queries)
        documents.extend(b_documents)
        y.extend(b_y)
    
    return qids, dids, queries, documents, y

def test_collate(batch):
    
    qids, dids, queries, documents, ys = [], [], [], [], []
    for (q, d, query, document, y) in batch:
        qids.append(q)
        dids.append(d)
        queries.append(query)
        documents.append(document)
        ys.append(y)
    
    return qids, dids, queries, documents, ys