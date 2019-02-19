import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import sys


AGG_OPS = ('none', 'maximum', 'minimum', 'count', 'sum', 'average')
class WordEmbedding(nn.Module):
    def __init__(self, word_emb, N_word, gpu, SQL_TOK,
            trainable=False):
        super(WordEmbedding, self).__init__()
        self.trainable = trainable
        self.N_word = N_word
        self.gpu = gpu
        self.SQL_TOK = SQL_TOK

        if trainable:
            print ("Using trainable embedding")
            self.w2i, word_emb_val = word_emb
            # tranable when using pretrained model, init embedding weights using prev embedding
            self.embedding = nn.Embedding(len(self.w2i), N_word)
            self.embedding.weight = nn.Parameter(torch.from_numpy(word_emb_val.astype(np.float32)))
        else:
            # else use word2vec or glove
            self.word_emb = word_emb
            print ("Using fixed embedding")

    def gen_x_q_batch(self, q):
        print("\n\ngen_x_q_batch STARTED")
        """


        q: list of tokenized queries - list of lists
        B: batch size
        val_embs:
        val_len: 
        """
        # print("\nq = " + str(q))
        B = len(q)
        # print("Batch size = " + str(B))
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, one_q in enumerate(q):
            """
            # for each sentence in the batch
            # i = (int) the index in the batch = 0 
            # one_q = (list) sentence in the batch like ['Please', 'show', 'the', 'most', 'common', 'occupation', 'of', 'players', '.']
            """

            # print("i = " + str(i))
            # print("one_q = " + str(one_q))
            q_val = []
            for ws in one_q:
                # for each word (ws = str) in the sentence
                # get the word embedding and append it to q_val
                # print("ws = " + str(ws))
                q_val.append(self.word_emb.get(ws, 
                                              np.zeros(self.N_word, dtype=np.float32)))                
            # print("q_val = " + str(q_val))

            #<BEG> and <END>
            # add two all zero arrays to the beginning and end
            # val_embs is the whole batch's sentences' word embeddings
            val_embs.append([np.zeros(self.N_word, dtype=np.float32)] + \
                                q_val + \
                                [np.zeros(self.N_word, dtype=np.float32)])  

            
            val_len[i] = 1 + len(q_val) + 1
        max_len = max(val_len)

        # print("val_len = " + str(val_len))
        # print("len val_embs = " + str(len(val_embs[0])))
        # print("val_embs = " + str(val_embs[0]))
        # print("max_len = " + str(max_len))

        # # for debugging
        # sys.exit("Error message")

        # create an empty matrix with 
        # size batch_size x max sentence length x word embedding size
        val_emb_array = np.zeros((B, max_len, self.N_word), dtype=np.float32)


        for i in range(B):
            for t in range(len(val_embs[i])):
                # print("\ntype(val_embs[i][t]) = " + str(type(val_embs[i][t])))
                # print("\nval_embs[i][t] = " + str(val_embs[i][t]))

                # print("len(val_embs[i][t]) = " + str(len(val_embs[i][t])))
                # print("list(val_embs[i][t]) = " + str(list(val_embs[i][t])))

                # why are there two of these?
                # val_emb_array[i, t, :] = list(val_embs[i][t])
                val_emb_array[i, t, :] = val_embs[i][t]
        val_inp = torch.from_numpy(val_emb_array)
        # print("val_inp = " + str(val_inp))

        if self.gpu:
            val_inp = val_inp.cuda()


        val_inp_var = Variable(val_inp)
        # print("val_inp_var = " + str(val_inp_var))

        # for debugging
        # sys.exit("Error message")


        return val_inp_var, val_len

    def gen_x_history_batch(self, history):
        print("gen_x_history_batch")
        B = len(history)
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, one_history in enumerate(history):
            history_val = []
            for item in one_history:
                #col
                if isinstance(item, list) or isinstance(item, tuple):
                    emb_list = []
                    ws = item[0].split() + item[1].split()
                    ws_len = len(ws)
                    for w in ws:
                        emb_list.append(self.word_emb.get(w, np.zeros(self.N_word, dtype=np.float32)))
                    if ws_len == 0:
                        raise Exception("word list should not be empty!")
                    elif ws_len == 1:
                        history_val.append(emb_list[0])
                    else:
                        history_val.append(sum(emb_list) / float(ws_len))
                #ROOT
                # elif isinstance(item,basestring):
                elif isinstance(item,str):
                    if item == "ROOT":
                        item = "root"
                    elif item == "asc":
                        item = "ascending"
                    elif item == "desc":
                        item == "descending"
                    if item in (
                    "none", "select", "from", "where", "having", "limit", "intersect", "except", "union", 'not',
                    'between', '=', '>', '<', 'in', 'like', 'is', 'exists', 'root', 'ascending', 'descending'):
                        history_val.append(self.word_emb.get(item, np.zeros(self.N_word, dtype=np.float32)))
                    elif item == "orderBy":
                        history_val.append((self.word_emb.get("order", np.zeros(self.N_word, dtype=np.float32)) +
                                            self.word_emb.get("by", np.zeros(self.N_word, dtype=np.float32))) / 2)
                    elif item == "groupBy":
                        history_val.append((self.word_emb.get("group", np.zeros(self.N_word, dtype=np.float32)) +
                                            self.word_emb.get("by", np.zeros(self.N_word, dtype=np.float32))) / 2)
                    elif item in ('>=', '<=', '!='):
                        history_val.append((self.word_emb.get(item[0], np.zeros(self.N_word, dtype=np.float32)) +
                                            self.word_emb.get(item[1], np.zeros(self.N_word, dtype=np.float32))) / 2)
                elif isinstance(item,int):
                    history_val.append(self.word_emb.get(AGG_OPS[item], np.zeros(self.N_word, dtype=np.float32)))
                else:
                    print("Warning: unsupported data type in history! {}".format(item))

            val_embs.append(history_val)
            val_len[i] = len(history_val)
        max_len = max(val_len)

        val_emb_array = np.zeros((B, max_len, self.N_word), dtype=np.float32)
        for i in range(B):
            for t in range(len(val_embs[i])):
                val_emb_array[i, t, :] = val_embs[i][t]
        val_inp = torch.from_numpy(val_emb_array)
        if self.gpu:
            val_inp = val_inp.cuda()

        val_inp_var = Variable(val_inp)
        print("val_inp_var = " + str(val_inp_var))

        # for debugging
        # sys.exit("Error message")

        return val_inp_var, val_len


    def gen_word_list_embedding(self,words,B):

        val_emb_array = np.zeros((B, len(words), self.N_word), dtype=np.float32)
        for i,word in enumerate(words):
            if len(word.split()) == 1:
                emb = self.word_emb.get(word, np.zeros(self.N_word, dtype=np.float32))
            else:
                word = word.split()
                emb = (self.word_emb.get(word[0], np.zeros(self.N_word, dtype=np.float32))
                                       +self.word_emb.get(word[1], np.zeros(self.N_word, dtype=np.float32)) )/2
            for b in range(B):
                val_emb_array[b,i,:] = emb
        val_inp = torch.from_numpy(val_emb_array)
        if self.gpu:
            val_inp = val_inp.cuda()
        val_inp_var = Variable(val_inp)
        return val_inp_var


    def gen_col_batch(self, cols):
        ret = []
        col_len = np.zeros(len(cols), dtype=np.int64)

        names = []
        for b, one_cols in enumerate(cols):
            names = names + one_cols
            col_len[b] = len(one_cols)
        #TODO: what is the diff bw name_len and col_len?
        name_inp_var, name_len = self.str_list_to_batch(names)
        return name_inp_var, name_len, col_len

    def str_list_to_batch(self, str_list):
        """get a list var of wemb of words in each column name in current bactch"""
        B = len(str_list)

        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, one_str in enumerate(str_list):
            if self.trainable:
                val = [self.w2i.get(x, 0) for x in one_str]
            else:
                val = [self.word_emb.get(x, np.zeros(
                    self.N_word, dtype=np.float32)) for x in one_str]
            val_embs.append(val)
            val_len[i] = len(val)
        max_len = max(val_len)

        if self.trainable:
            val_tok_array = np.zeros((B, max_len), dtype=np.int64)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_tok_array[i,t] = val_embs[i][t]
            val_tok = torch.from_numpy(val_tok_array)
            if self.gpu:
                val_tok = val_tok.cuda()
            val_tok_var = Variable(val_tok)
            val_inp_var = self.embedding(val_tok_var)
        else:
            val_emb_array = np.zeros(
                    (B, max_len, self.N_word), dtype=np.float32)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_emb_array[i,t,:] = val_embs[i][t]
            val_inp = torch.from_numpy(val_emb_array)
            if self.gpu:
                val_inp = val_inp.cuda()
            val_inp_var = Variable(val_inp)

        return val_inp_var, val_len
