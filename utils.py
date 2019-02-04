import re
import io
import json
import numpy as np
import os
import signal
from preprocess_train_dev_data import get_table_dict
from data.process_sql import tokenize


def load_train_dev_dataset(component,train_dev,history, root):
    """
    ex: 
    """
    print("Loading dataset = {}/{}_{}_{}_dataset.json".format(root,history,
                                                            train_dev,component))
    return json.load(open("{}/{}_{}_{}_dataset.json".format(root, 
                                                            history,
                                                            train_dev,
                                                            component)))


def to_batch_seq(data, idxes, st, ed):
    # to_batch_seq(data, perm, st, ed)
    q_seq = []
    history = []
    label = []
    for i in range(st, ed):
        q_seq.append(data[idxes[i]]['question_tokens'])
        history.append(data[idxes[i]]["history"])
        label.append(data[idxes[i]]["label"])
    return q_seq,history,label



# CHANGED
def to_batch_tables(data, idxes, st,ed, table_type):
    # col_lens = []
    col_seq = []
    for i in range(st, ed):
        ts = data[idxes[i]]["ts"]
        tname_toks = [x.split(" ") for x in ts[0]]
        col_type = ts[2]
        cols = [x.split(" ") for xid, x in ts[1]]
        tab_seq = [xid for xid, x in ts[1]]
        cols_add = []
        for tid, col, ct in zip(tab_seq, cols, col_type):
            col_one = [ct]
            if tid == -1:
                tabn = ["all"]
            else:
                if table_type=="no": tabn = []
                else: tabn = tname_toks[tid]
            for t in tabn:
                if t not in col:
                    col_one.append(t)
            col_one.extend(col)
            cols_add.append(col_one)
        col_seq.append(cols_add)

    return col_seq

## used for training in train.py
def epoch_train(model, optimizer, batch_size, component, embed_layer, data, table_type):
    """
    epoch_train(model,  -> one of MultiSqlPredictor, KeyWordPredictor, ColPredictor
                            OpPredictor, AggPredictor, RootTeminalPredictor, 
                            DesAscLimitPredictor, HavingPredictor, AndOrPredictor
                optimizer, -> torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0)
                BATCH_SIZE, -> 20 if toy, 64 if normal
                args.train_component, -> one of train components,available:[multi_sql,keyword,col,op,agg,root_tem,des_asc,having,andor]'
                embed_layer, -> WordEmbedding(word_emb, 
                                N_word, 
                                gpu=GPU,
                                SQL_TOK=SQL_TOK, 
                                trainable=args.train_emb)
                train_data, -> load_train_dev_dataset(args.train_component, "train", 
                                        args.history_type, 
                                        args.data_root)
                table_type=args.table_type -> choices=['std','no']
                                            standard, hierarchical, or no table info
                                            used only in: col,op,agg,root_tem,des_asc,having modules
    """

    model.train()
    perm = np.random.permutation(len(data))
    cum_loss = 0.0
    st = 0

    while st < len(data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)
        # print("ed = {}".format(ed))
        q_seq, history,label = to_batch_seq(data, perm, st, ed)

        # print("q_seq = {}".format(q_seq))
        q_emb_var, q_len = embed_layer.gen_x_q_batch(q_seq)

        hs_emb_var, hs_len = embed_layer.gen_x_history_batch(history)
        score = 0.0
        loss = 0.0

        if component == "multi_sql":
            # trained by Cross Entropy
            mkw_emb_var = embed_layer.gen_word_list_embedding(["none","except","intersect","union"],(ed-st))
            mkw_len = np.full(q_len.shape, 4,dtype=np.int64)
            # print("mkw_emb:{}".format(mkw_emb_var.size()))
            score = model.forward(q_emb_var, 
                                q_len,
                                hs_emb_var, 
                                hs_len, 
                                mkw_emb_var=mkw_emb_var, 
                                mkw_len=mkw_len)
        
        elif component == "keyword":
            # 
            # where group by order by
            # [[0,1,2]]
            kw_emb_var = embed_layer.gen_word_list_embedding(["where", "group by", "order by"],(ed-st))
            mkw_len = np.full(q_len.shape, 3, dtype=np.int64)
            score = model.forward(q_emb_var, 
                                    q_len, 
                                    hs_emb_var, 
                                    hs_len, 
                                    kw_emb_var=kw_emb_var, 
                                    kw_len=mkw_len)
        
        elif component == "col":
            #col word embedding
            # [[0,1,3]]
            col_seq = to_batch_tables(data, perm, st, ed, table_type)
            col_emb_var, col_name_len, col_len = embed_layer.gen_col_batch(col_seq)
            score = model.forward(q_emb_var, 
                                    q_len, 
                                    hs_emb_var, 
                                    hs_len, 
                                    col_emb_var, 
                                    col_len, 
                                    col_name_len)

        elif component == "op":
            #B*index
            gt_col = np.zeros(q_len.shape,dtype=np.int64)
            index = 0
            for i in range(st,ed):
                # print(i)
                gt_col[index] = data[perm[i]]["gt_col"]
                index += 1

            col_seq = to_batch_tables(data, perm, st, ed, table_type)
            col_emb_var, col_name_len, col_len = embed_layer.gen_col_batch(col_seq)
            score = model.forward(q_emb_var, 
                                    q_len, 
                                    hs_emb_var, 
                                    hs_len, 
                                    col_emb_var, 
                                    col_len, 
                                    col_name_len, 
                                    gt_col=gt_col)

        elif component == "agg":
            # [[0,1,3]]
            col_seq = to_batch_tables(data, perm, st, ed, table_type)
            col_emb_var, col_name_len, col_len = embed_layer.gen_col_batch(col_seq)
            gt_col = np.zeros(q_len.shape, dtype=np.int64)
            # print(ed)
            index = 0
            for i in range(st, ed):
                # print(i)
                gt_col[index] = data[perm[i]]["gt_col"]
                index += 1
            score = model.forward(q_emb_var, 
                                    q_len, 
                                    hs_emb_var, 
                                    hs_len, 
                                    col_emb_var, 
                                    col_len, 
                                    col_name_len, 
                                    gt_col=gt_col)

        elif component == "root_tem":
            #B*0/1
            col_seq = to_batch_tables(data, perm, st, ed, table_type)
            col_emb_var, col_name_len, col_len = embed_layer.gen_col_batch(col_seq)
            gt_col = np.zeros(q_len.shape, dtype=np.int64)
            # print(ed)
            index = 0
            for i in range(st, ed):
                # print(data[perm[i]]["history"])
                gt_col[index] = data[perm[i]]["gt_col"]
                index += 1
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, gt_col=gt_col)

        elif component == "des_asc":
            # B*0/1
            col_seq = to_batch_tables(data, perm, st, ed, table_type)
            col_emb_var, col_name_len, col_len = embed_layer.gen_col_batch(col_seq)
            gt_col = np.zeros(q_len.shape, dtype=np.int64)
            # print(ed)
            index = 0
            for i in range(st, ed):
                # print(i)
                gt_col[index] = data[perm[i]]["gt_col"]
                index += 1
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, gt_col=gt_col)

        elif component == 'having':
            col_seq = to_batch_tables(data, perm, st, ed, table_type)
            col_emb_var, col_name_len, col_len = embed_layer.gen_col_batch(col_seq)
            gt_col = np.zeros(q_len.shape, dtype=np.int64)
            # print(ed)
            index = 0
            for i in range(st, ed):
                # print(i)
                gt_col[index] = data[perm[i]]["gt_col"]
                index += 1
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, gt_col=gt_col)

        elif component == "andor":
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len)
        
        # score = model.forward(q_seq, col_seq, col_num, pred_entry,
        #         gt_where=gt_where_seq, gt_cond=gt_cond_seq, gt_sel=gt_sel_seq)
        # print("label {}".format(label))
        loss = model.loss(score, label)
        print("loss {}".format(loss.data.cpu().numpy()))
        # cum_loss += loss.data.cpu().numpy()[0]*(ed - st)
        cum_loss += loss.data.cpu().numpy()*(ed - st)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        st = ed

    return cum_loss / len(data)

## used for development evaluation in train.py
def epoch_acc(model, batch_size, component, embed_layer, data, table_type, error_print=False, train_flag = False):
    
    model.eval()
    perm = list(range(len(data)))
    st = 0
    total_number_error = 0.0
    total_p_error = 0.0
    total_error = 0.0
    print("dev data size {}".format(len(data)))
    while st < len(data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)
        # print("ed: {}".format(ed))

        q_seq, history, label = to_batch_seq(data, perm, st, ed)
        q_emb_var, q_len = embed_layer.gen_x_q_batch(q_seq)
        hs_emb_var, hs_len = embed_layer.gen_x_history_batch(history)
        score = 0.0

        if component == "multi_sql":
            # none, except, intersect, union
            # truth B*index(0,1,2,3)
            # print("hs_len:{}".format(hs_len))
            # print("q_emb_shape:{} hs_emb_shape:{}".format(q_emb_var.size(), hs_emb_var.size()))
            mkw_emb_var = embed_layer.gen_word_list_embedding(["none","except","intersect","union"],(ed-st))
            mkw_len = np.full(q_len.shape, 4,dtype=np.int64)
            # print("mkw_emb:{}".format(mkw_emb_var.size()))
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len, mkw_emb_var=mkw_emb_var, mkw_len=mkw_len)
            # print("score: {}".format(score))

        elif component == "keyword":
            #where group by order by
            # [[0,1,2]]
            kw_emb_var = embed_layer.gen_word_list_embedding(["where", "group by", "order by"],(ed-st))
            mkw_len = np.full(q_len.shape, 3, dtype=np.int64)
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len, kw_emb_var=kw_emb_var, kw_len=mkw_len)
        
        elif component == "col":
            #col word embedding
            # [[0,1,3]]
            col_seq = to_batch_tables(data, perm, st, ed, table_type)
            col_emb_var, col_name_len, col_len = embed_layer.gen_col_batch(col_seq)
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len)
        
        elif component == "op":
            #B*index
            col_seq = to_batch_tables(data, perm, st, ed, table_type)
            col_emb_var, col_name_len, col_len = embed_layer.gen_col_batch(col_seq)
            gt_col = np.zeros(q_len.shape,dtype=np.int64)
            # print(ed)
            index = 0
            for i in range(st,ed):
                # print(i)
                gt_col[index] = data[perm[i]]["gt_col"]
                index += 1
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, gt_col=gt_col)

        elif component == "agg":
            # [[0,1,3]]
            col_seq = to_batch_tables(data, perm, st, ed, table_type)
            col_emb_var, col_name_len, col_len = embed_layer.gen_col_batch(col_seq)
            gt_col = np.zeros(q_len.shape, dtype=np.int64)
            # print(ed)
            index = 0
            for i in range(st, ed):
                # print(i)
                gt_col[index] = data[perm[i]]["gt_col"]
                index += 1

            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, gt_col=gt_col)

        elif component == "root_tem":
            #B*0/1
            col_seq = to_batch_tables(data, perm, st, ed, table_type)
            col_emb_var, col_name_len, col_len = embed_layer.gen_col_batch(col_seq)
            gt_col = np.zeros(q_len.shape, dtype=np.int64)
            # print(ed)
            index = 0
            for i in range(st, ed):
                # print(data[perm[i]]["history"])
                gt_col[index] = data[perm[i]]["gt_col"]
                index += 1
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, gt_col=gt_col)

        elif component == "des_asc":
            # B*0/1
            col_seq = to_batch_tables(data, perm, st, ed, table_type)
            col_emb_var, col_name_len, col_len = embed_layer.gen_col_batch(col_seq)
            gt_col = np.zeros(q_len.shape, dtype=np.int64)
            # print(ed)
            index = 0
            for i in range(st, ed):
                # print(i)
                gt_col[index] = data[perm[i]]["gt_col"]
                index += 1
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, gt_col=gt_col)

        elif component == 'having':
            col_seq = to_batch_tables(data, perm, st, ed, table_type)
            col_emb_var, col_name_len, col_len = embed_layer.gen_col_batch(col_seq)
            gt_col = np.zeros(q_len.shape, dtype=np.int64)
            # print(ed)
            index = 0
            for i in range(st, ed):
                # print(i)
                gt_col[index] = data[perm[i]]["gt_col"]
                index += 1
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, gt_col=gt_col)

        elif component == "andor":
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len)
        # print("label {}".format(label))

        if component in ("agg","col","keyword","op"):
            num_err, p_err, err = model.check_acc(score, label)
            total_number_error += num_err
            total_p_error += p_err
            total_error += err
        else:
            err = model.check_acc(score, label)
            total_error += err
            # print("err: {}, total_error: {}".format(err,total_error))

        st = ed

    if component in ("agg","col","keyword","op"):
        print("Dev {} acc number predict acc:{} partial acc: {} total acc: {}".format(component,1 - total_number_error*1.0/len(data),1 - total_p_error*1.0/len(data),  1 - total_error*1.0/len(data)))
        return 1 - total_error*1.0/len(data)
    else:
        print("Dev {} acc total acc: {}".format(component,1 - total_error*1.0/len(data)))
        return 1 - total_error*1.0/len(data)


def timeout_handler(num, stack):
    print("Received SIGALRM")
    raise Exception("Timeout")



## used in test.py
def test_acc(model, batch_size, data, output_path):

    """
    works with: python test.py --test_data_path data/spider/dev.json
    doesn't work with the full_dev_multi_sql_dataset.json like data
    No evaluation criteria
    """
    table_dict = get_table_dict("./data/spider/tables.json")
    # print("\ntable_dict keys = " + str(table_dict.keys()))
    # print("\ntable_dict[perpetrator] = " + str(table_dict["perpetrator"]))
    # print("\ntable_dict[perpetrator] keys = " + str(table_dict["perpetrator"].keys()))
    # print("\ntable_dict[perpetrator][db_id] = " + str(table_dict["perpetrator"]["db_id"]))
    # print("\noutput path = " + str(output_path))
    # print("\ntype(data) = " + str(type(data)))
    # print("data[1] = " + str(data[1]))

    f = open(output_path,"w")

    for item in data[:]:
        
        # print("\nitem = " + str(item))
        db_id = item["db_id"]
        if db_id not in table_dict: print("\nError %s not in table_dict" % db_id)
        
        # signal.signal(signal.SIGALRM, timeout_handler)
        # signal.alarm(2) # set timer to prevent infinite recursion in SQL generation
        
        # print("\nitem['question_toks']]*batch_size = " + str([item["question_toks"]]*batch_size))
        # print("\ntable_dict[db_id] = " + str(table_dict[db_id]))
        
        sql = model.forward([item["question_toks"]]*batch_size,
                            [],
                            table_dict[db_id])
        
        if sql is not None:
            # print(sql)
            sql = model.gen_sql(sql,table_dict[db_id])
        else:
            sql = "select a from b"

        # print("Generated sql = " + str(sql))
        # print("")
        f.write("{}\n".format(sql))

    f.close()


## used in inference.py
def infer_sql(model, batch_size, nlq, table_dict, output_path):

    """
    works with: python test.py --test_data_path data/spider/dev.json
    doesn't work with the full_dev_multi_sql_dataset.json like data

    item = {'db_id': 'department_management', 
            'query': 'SELECT max(budget_in_billions) ,  min(budget_in_billions) FROM department', 
            'query_toks': ['SELECT', 'max', '(', 'budget_in_billions', ')', ',', 'min', '(', 'budget_in_billions', ')', 'FROM', 'department'], 
            'query_toks_no_value': ['select', 'max', '(', 'budget_in_billions', ')', ',', 'min', '(', 'budget_in_billions', ')', 'from', 'department'], 
            'question': 'What are the maximum and minimum budget of the departments?', 
            'question_toks': ['What', 'are', 'the', 'maximum', 'and', 'minimum', 'budget', 'of', 'the', 'departments', '?'], 
            'sql': {'except': None, 
                    'from': {'conds': [], 'table_units': [['table_unit', 0]]}, 
                    'groupBy': [], 'having': [], 
                    'intersect': None, 
                    'limit': None, 
                    'orderBy': [], 
                    'select': [False, [[1, [0, [0, 5, False], None]], [2, [0, [0, 5, False], None]]]], 
                    'union': None, 'where': []}}

    The user should give the natural language question + info about:
    table_dict[db_id] = 
            {'column_names': [[-1, '*'], [0, 'department id'], [0, 'name'], [0, 'creation'], [0, 'ranking'], [0, 'budget in billions'], [0, 'num employees'], [1, 'head id'], [1, 'name'], [1, 'born state'], [1, 'age'], [2, 'department id'], [2, 'head id'], [2, 'temporary acting']], 
            'column_names_original': [[-1, '*'], [0, 'Department_ID'], [0, 'Name'], [0, 'Creation'], [0, 'Ranking'], [0, 'Budget_in_Billions'], [0, 'Num_Employees'], [1, 'head_ID'], [1, 'name'], [1, 'born_state'], [1, 'age'], [2, 'department_ID'], [2, 'head_ID'], [2, 'temporary_acting']], 
            'column_types': ['text', 'number', 'text', 'text', 'number', 'number', 'number', 'number', 'text', 'text', 'number', 'number', 'number', 'text'], 
            'db_id': 'department_management', 
            'foreign_keys': [[12, 7], [11, 1]], 
            'primary_keys': [1, 7, 11], 
            'table_names': ['department', 'head', 'management'], 
            'table_names_original': ['department', 'head', 'management']}

    [item["question_toks"]] = 
            [['What', 'are', 'the', 'distinct', 'creation', 'years', 'of', 
            'the', 'departments', 'managed', 'by', 'a', 'secretary', 'born',
             'in', 'state', "'Alabama", "'", '?']]  

        gen_sql = infer_sql(model = model, 
                        batch_size = BATCH_SIZE, 
                        nlq = nlq, 
                        table_dict = table_dict,
                        output_path = OUTPUT_PATH)

    """

    item = nlq
    db_id = item["db_id"]


    # table_dict = get_table_dict("./data/spider/tables.json")
    # print("\ntable_dict keys = " + str(table_dict.keys()))
    # print("\ntable_dict[perpetrator] = " + str(table_dict["perpetrator"]))
    # print("\ntable_dict[perpetrator] keys = " + str(table_dict["perpetrator"].keys()))
    # print("\ntable_dict[perpetrator][db_id] = " + str(table_dict["perpetrator"]["db_id"]))
    # print("\noutput path = " + str(output_path))
    # print("\ntype(data) = " + str(type(data)))
    # print("data[1] = " + str(data[1]))

    f = open(output_path,"w")

    # for item in data[:]:
        
    # item = data    
    print("\nitem = " + str(item))

    if db_id not in table_dict: print("\nError %s not in table_dict" % db_id)
    
    # signal.signal(signal.SIGALRM, timeout_handler)
    # signal.alarm(2) # set timer to prevent infinite recursion in SQL generation
    
    # print("item['question_toks']]*batch_size = " + str([item["question_toks"]]*batch_size))
    # print("item['question_toks']] = " + str([item["question_toks"]]))
    # print("table_dict[db_id] = " + str(table_dict[db_id]))
    
    # sql = model.forward([item["question_toks"]]*batch_size,
    sql = model.forward([item["question_toks"]],
                        [],
                        table_dict[db_id])
    
    if sql is not None:
        print(sql)
        sql = model.gen_sql(sql,
                            table_dict[db_id])
    else:
        sql = "select a from b"
    print(sql)
    print("")
    f.write("\n")  # new line to seperate queries
    f.write("{}\n".format(sql))

    f.close()

    return sql


def load_word_emb(file_name, load_used=False, use_small=False):
    """
    Used to load the word embeddings like Glove or Word2Vec

    Used like: load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word), 
                                load_used=args.train_emb, 
                                use_small=USE_SMALL)
    """
    if not load_used:
        print ('Loading word embedding from %s'%file_name)
        ret = {}
        with open(file_name) as inf:
            for idx, line in enumerate(inf):
                if (use_small and idx >= 5000):
                    break
                # print("\nline = " + str(line)) # line = visa -0.22659 0.82905 -0.38674 0.14165 0.4468 0.36279
                info = line.strip().split(' ')
                # print("info = " + str(info))   # info = ['visa', '-0.22659', '0.82905', '-0.38674', '0.14165',
                if info[0].lower() not in ret:
                    # if the word is not in the dictionary 'ret', add it
                    # word_embedding = np.array(map(lambda x:float(x), info[1:]))
                    word_embedding = np.array(list(map(lambda x:float(x), info[1:])))
                    # print("word_embedding = " + str(word_embedding))
                    ret[info[0]] = word_embedding
        return ret
    else:
        print ('Load used word embedding')
        with open('./glove/word2idx.json') as inf:
            w2i = json.load(inf)
        with open('./glove/usedwordemb.npy') as inf:
            word_emb_val = np.load(inf)
        
        return w2i, word_emb_val


import sqlite3
import pandas as pd

def get_table_names(conn):
    res = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = []
    table_names.append("na") # for some reason, you have to add one element before titles
    for name in res:
#         print(name[0])
        table_names.append(name[0])
    return table_names

def get_tables_html(db_name):
    
    conn = sqlite3.connect("./data/spider/database/{}/{}.sqlite".format(db_name,db_name))
    table_names_db = get_table_names(conn = conn)
    
    table_html = []
    for i in table_names_db:
        if i != "na":
            df = pd.read_sql_query("select * from {} limit 3;".format(i), conn)
            print(df)
            table_html.append(df.to_html())
    return table_html, table_names_db


from preprocess_train_dev_data import *

## used for training based on user fedback in train_feedback.py
def epoch_feedback_train(model, optimizer, batch_size, component, 
                        embed_layer, data, table_type, nlq, db_name, 
                        correct_query, correct_query_data):
    """
    Select a random batch (size = batch size + 1)
    Add the feedback query and language
    """   
    

    optimizer.zero_grad()
    model.train()
    perm = np.random.permutation(len(data))
    cum_loss = 0.0
    st = 0

    # while st < len(data):

    # ed = st+batch_size if st+batch_size < len(perm) else len(perm)
    # print("ed = {}".format(ed))
    ed = batch_size - 1
    q_seq, history,label = to_batch_seq(data, perm, st, ed)
    # q_seq, history, label are all lists
    # print("q_seq, type = {}, {}".format(q_seq, type(q_seq)))
    # print("history, type = {}, {}".format(history, type(history)))
    # print("label, type = {}, {}".format(label, type(label)))      # loss = model.loss(score, label)

    # add the correct query given by the user
    # print("db_id: {}".format(db_name)) 
    # print("query: {}".format(correct_query)) 
    # print("query_toks: {}".format(tokenize(correct_query))) 
    # print("question: {}".format(nlq))
    # print("question_toks: {}".format(tokenize(nlq)))
    # print("sql: {}".format())
    
    # component = "col"
    name_dataset = component + "_dataset"
    # print("correct query dataset: {}".format(correct_query_data[name_dataset]))
    
    # if correct_query_data[name_dataset] != []:
    #     print("correct query question_tokens: {}".format(correct_query_data[name_dataset][0]["question_tokens"]))
    #     print("correct query history: {}".format(correct_query_data[name_dataset][0]["history"]))
    #     print("correct query label: {}".format(correct_query_data[name_dataset][0]["label"]))

    # exit if there is no component to train on
    if correct_query_data[name_dataset] == []:
        print("NOTHING TO TRAIN ON FOR COMPONENT: {}".format(component))
        return 

    # add to original dataset
    q_seq.append(correct_query_data[name_dataset][0]["question_tokens"])
    history.append(correct_query_data[name_dataset][0]["history"])
    label.append(correct_query_data[name_dataset][0]["label"])

    # if True:
    #     return

    q_emb_var, q_len = embed_layer.gen_x_q_batch(q_seq)

    hs_emb_var, hs_len = embed_layer.gen_x_history_batch(history)

    score = 0.0
    loss = 0.0

    # fix ed after correct query addition
    ed = batch_size

    if component == "multi_sql":
        # trained by Cross Entropy
        mkw_emb_var = embed_layer.gen_word_list_embedding(["none","except","intersect","union"],(ed-st))
        mkw_len = np.full(q_len.shape, 4,dtype=np.int64)
        # print("mkw_emb:{}".format(mkw_emb_var.size()))
        score = model.forward(q_emb_var, 
                            q_len,
                            hs_emb_var, 
                            hs_len, 
                            mkw_emb_var=mkw_emb_var, 
                            mkw_len=mkw_len)
    
    elif component == "keyword":
        # 
        # where group by order by
        # [[0,1,2]]
        kw_emb_var = embed_layer.gen_word_list_embedding(["where", "group by", "order by"],(ed-st))
        mkw_len = np.full(q_len.shape, 3, dtype=np.int64)
        score = model.forward(q_emb_var, 
                                q_len, 
                                hs_emb_var, 
                                hs_len, 
                                kw_emb_var=kw_emb_var, 
                                kw_len=mkw_len)
    
    elif component == "col":
        #col word embedding
        # [[0,1,3]]
        col_seq = to_batch_tables(data, perm, st, ed, table_type)
        col_emb_var, col_name_len, col_len = embed_layer.gen_col_batch(col_seq)
        score = model.forward(q_emb_var, 
                                q_len, 
                                hs_emb_var, 
                                hs_len, 
                                col_emb_var, 
                                col_len, 
                                col_name_len)

    elif component == "op":
        #B*index
        gt_col = np.zeros(q_len.shape,dtype=np.int64)
        index = 0
        for i in range(st,ed):
            # print(i)
            gt_col[index] = data[perm[i]]["gt_col"]
            index += 1

        col_seq = to_batch_tables(data, perm, st, ed, table_type)
        col_emb_var, col_name_len, col_len = embed_layer.gen_col_batch(col_seq)
        score = model.forward(q_emb_var, 
                                q_len, 
                                hs_emb_var, 
                                hs_len, 
                                col_emb_var, 
                                col_len, 
                                col_name_len, 
                                gt_col=gt_col)

    elif component == "agg":
        # [[0,1,3]]
        col_seq = to_batch_tables(data, perm, st, ed, table_type)
        col_emb_var, col_name_len, col_len = embed_layer.gen_col_batch(col_seq)
        gt_col = np.zeros(q_len.shape, dtype=np.int64)
        # print(ed)
        index = 0
        for i in range(st, ed):
            # print(i)
            gt_col[index] = data[perm[i]]["gt_col"]
            index += 1
        score = model.forward(q_emb_var, 
                                q_len, 
                                hs_emb_var, 
                                hs_len, 
                                col_emb_var, 
                                col_len, 
                                col_name_len, 
                                gt_col=gt_col)

    elif component == "root_tem":
        #B*0/1
        col_seq = to_batch_tables(data, perm, st, ed, table_type)
        col_emb_var, col_name_len, col_len = embed_layer.gen_col_batch(col_seq)
        gt_col = np.zeros(q_len.shape, dtype=np.int64)
        # print(ed)
        index = 0
        for i in range(st, ed):
            # print(data[perm[i]]["history"])
            gt_col[index] = data[perm[i]]["gt_col"]
            index += 1
        score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, gt_col=gt_col)

    elif component == "des_asc":
        # B*0/1
        col_seq = to_batch_tables(data, perm, st, ed, table_type)
        col_emb_var, col_name_len, col_len = embed_layer.gen_col_batch(col_seq)
        gt_col = np.zeros(q_len.shape, dtype=np.int64)
        # print(ed)
        index = 0
        for i in range(st, ed):
            # print(i)
            gt_col[index] = data[perm[i]]["gt_col"]
            index += 1
        score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, gt_col=gt_col)

    elif component == 'having':
        col_seq = to_batch_tables(data, perm, st, ed, table_type)
        col_emb_var, col_name_len, col_len = embed_layer.gen_col_batch(col_seq)
        gt_col = np.zeros(q_len.shape, dtype=np.int64)
        # print(ed)
        index = 0
        for i in range(st, ed):
            # print(i)
            gt_col[index] = data[perm[i]]["gt_col"]
            index += 1
        score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, gt_col=gt_col)

    elif component == "andor":
        score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len)
    
    # score = model.forward(q_seq, col_seq, col_num, pred_entry,
    #         gt_where=gt_where_seq, gt_cond=gt_cond_seq, gt_sel=gt_sel_seq)
    # print("label {}".format(label))
    loss = model.loss(score, label)
    print("loss {}".format(loss.data.cpu().numpy()))
    # cum_loss += loss.data.cpu().numpy()[0]*(ed - st)
    cum_loss += loss.data.cpu().numpy()*(ed - st)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    st = ed

    return cum_loss / len(data)
