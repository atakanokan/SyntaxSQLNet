"""
This script is for training the model further with the user supplied correct query to
make the model incrementally better.
"""

import json
import torch
import datetime
import argparse
import numpy as np
import random
from utils import *
from word_embedding import WordEmbedding
from models.agg_predictor import AggPredictor
from models.col_predictor import ColPredictor
from models.desasc_limit_predictor import DesAscLimitPredictor
from models.having_predictor import HavingPredictor
from models.keyword_predictor import KeyWordPredictor
from models.multisql_predictor import MultiSqlPredictor
from models.op_predictor import OpPredictor
from models.root_teminal_predictor import RootTeminalPredictor
from models.andor_predictor import AndOrPredictor

from preprocess_train_dev_data import get_table_dict
from data.process_sql import *
from data.parse_sql_one import *
from data.process_sql import tokenize
from inference import infer_sql, infer_script


TRAIN_COMPONENTS = ('multi_sql','keyword','col','op','agg','root_tem','des_asc','having','andor')
SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND', 'EQL', 'GT', 'LT', '<BEG>']

def train_feedback(nlq,db_name,correct_query,toy,word_emb):
    """
    Arguments:
        nlq: english question (tokenization is done here) - get from Flask (User)
        db_name: name of the database the query targets - get from Flask (User)
        correct_query: the ground truth query supplied by the user(s) - get from Flask
        toy: uses a small example of word embeddings to debug faster
    """

    ITER = 21

    SAVED_MODELS_FOLDER = "saved_models"
    OUTPUT_PATH = "output_inference.txt"
    HISTORY_TYPE = "full"
    GPU_ENABLE = False
    TRAIN_EMB = False
    TABLE_TYPE = "std"
    DATA_ROOT = "generated_data"

    use_hs = True
    if HISTORY_TYPE == "no":
        HISTORY_TYPE = "full"
        use_hs = False

    """
    Model Hyperparameters
    """
    N_word=300     # word embedding dimension
    B_word=42      # 42B tokens in the Glove pretrained embeddings
    N_h = 300      # hidden size dimension
    N_depth=2      # 


    if toy:
        USE_SMALL=True
        # GPU=True
        GPU = GPU_ENABLE
        BATCH_SIZE=20
    else:
        USE_SMALL=False
        # GPU=True
        GPU = GPU_ENABLE
        BATCH_SIZE=64
    # TRAIN_ENTRY=(False, True, False)  # (AGG, SEL, COND)
    # TRAIN_AGG, TRAIN_SEL, TRAIN_COND = TRAIN_ENTRY
    learning_rate = 1e-4
    
    # GENERATE CORRECT QUERY DATASET
    table_data_path = "./data/spider/tables.json"
    table_dict = get_table_dict(table_data_path)
    train_data_path = "./data/spider/train_spider.json"
    train_data = json.load(open(train_data_path))
    sql = correct_query             #"SELECT name ,  country ,  age FROM singer ORDER BY age DESC"
    db_id = db_name                 #"concert_singer"
    table_file = table_data_path    # "tables.json"

    schemas, db_names, tables = get_schemas_from_json(table_file)
    schema = schemas[db_id]
    table = tables[db_id]
    schema = Schema(schema, table)
    sql_label = get_sql(schema, sql)
    correct_query_data = {
        "multi_sql_dataset": [],
        "keyword_dataset": [],
        "col_dataset": [],
        "op_dataset": [],
        "agg_dataset": [],
        "root_tem_dataset": [],
        "des_asc_dataset": [],
        "having_dataset": [],
        "andor_dataset":[]
    }
    parser_item_with_long_history(tokenize(nlq), #item["question_toks"], 
                              sql_label,  #item["sql"], 
                              table_dict[db_name],  #table_dict[item["db_id"]], 
                                [], 
                                correct_query_data)
    # print("\nCorrect query dataset: {}".format(correct_query_data))



    for train_component in TRAIN_COMPONENTS:
        print("\nTRAIN COMPONENT: {}".format(train_component))
        # Check if the compenent to be trained is an actual component
        if train_component not in TRAIN_COMPONENTS:
            print("Invalid train component")
            exit(1)


        """
        Read in the data
        """
        train_data = load_train_dev_dataset(train_component, 
                                            "train", 
                                            HISTORY_TYPE, 
                                            DATA_ROOT)
        # print("train_data type: {}".format(type(train_data)))
        dev_data = load_train_dev_dataset(train_component, "dev", 
                                            HISTORY_TYPE, 
                                            DATA_ROOT)
        # sql_data, table_data, val_sql_data, val_table_data, \
        #         test_sql_data, test_table_data, \
        #         TRAIN_DB, DEV_DB, TEST_DB = load_dataset(args.dataset, use_small=USE_SMALL)

        if GPU_ENABLE:
            map_to = "gpu"
        else:
            map_to = "cpu"
        
        # Selecting which Model to Train
        model = None
        if train_component == "multi_sql":
            model = MultiSqlPredictor(N_word=N_word,  
                                    N_h=N_h,
                                    N_depth=N_depth,
                                    gpu=GPU, 
                                    use_hs=use_hs)
            model.load_state_dict(torch.load("{}/multi_sql_models.dump".format(SAVED_MODELS_FOLDER),map_location=map_to))
    
        elif train_component == "keyword":
            model = KeyWordPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,
                                    gpu=GPU, use_hs=use_hs)
            model.load_state_dict(torch.load("{}/keyword_models.dump".format(SAVED_MODELS_FOLDER),map_location=map_to))
    
        elif train_component == "col":
            model = ColPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,
                                gpu=GPU, use_hs=use_hs)
            model.load_state_dict(torch.load("{}/col_models.dump".format(SAVED_MODELS_FOLDER),map_location=map_to))
    
        elif train_component == "op":
            model = OpPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,
                                gpu=GPU, use_hs=use_hs)
            model.load_state_dict(torch.load("{}/op_models.dump".format(SAVED_MODELS_FOLDER),map_location=map_to))
                            
        elif train_component == "agg":
            model = AggPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,
                                gpu=GPU, use_hs=use_hs)
            model.load_state_dict(torch.load("{}/agg_models.dump".format(SAVED_MODELS_FOLDER),map_location=map_to))
    
        elif train_component == "root_tem":
            model = RootTeminalPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,
                                            gpu=GPU, use_hs=use_hs)
            model.load_state_dict(torch.load("{}/root_tem_models.dump".format(SAVED_MODELS_FOLDER),map_location=map_to))
    
        elif train_component == "des_asc":
            model = DesAscLimitPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,
                                        gpu=GPU, use_hs=use_hs)
            model.load_state_dict(torch.load("{}/des_asc_models.dump".format(SAVED_MODELS_FOLDER),map_location=map_to))
    
        elif train_component == "having":
            model = HavingPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,
                                    gpu=GPU, use_hs=use_hs)
            model.load_state_dict(torch.load("{}/having_models.dump".format(SAVED_MODELS_FOLDER),map_location=map_to))

        elif train_component == "andor":
            model = AndOrPredictor(N_word=N_word, N_h=N_h, N_depth=N_depth, 
                                    gpu=GPU, use_hs=use_hs)
            model.load_state_dict(torch.load("{}/andor_models.dump".format(SAVED_MODELS_FOLDER),map_location=map_to))


        # model = SQLNet(word_emb, N_word=N_word, gpu=GPU, trainable_emb=args.train_emb)
        
        optimizer = torch.optim.Adam(model.parameters(), 
                                    lr=learning_rate, 
                                    weight_decay = 0)
        print("finished build model")

        print_flag = False
        embed_layer = WordEmbedding(word_emb, 
                                    N_word, 
                                    gpu=GPU,
                                    SQL_TOK=SQL_TOK, 
                                    trainable=TRAIN_EMB)
        


        print("start training")
        best_acc = 0.0
        for i in range(ITER):
            print('ITER %d @ %s'%(i+1, datetime.datetime.now()))
            # arguments of epoch_train
            # model, optimizer, batch_size, component,embed_layer,data, table_type
            # print(' Loss = %s' % epoch_train(
            #                     model, optimizer, BATCH_SIZE,
            #                     args.train_component,
            #                     embed_layer,
            #                     train_data,
            #                     table_type=args.table_type))
            print('Total Loss = %s' % epoch_feedback_train(model = model, 
                                                    optimizer = optimizer, 
                                                    batch_size = BATCH_SIZE, 
                                                    component = train_component, 
                                                    embed_layer = embed_layer, 
                                                    data = train_data, 
                                                    table_type = TABLE_TYPE,
                                                    nlq = nlq, 
                                                    db_name = db_name, 
                                                    correct_query = correct_query,
                                                    correct_query_data = correct_query_data))
            
            # Check improvement every 10 iterations
            if i % 10 == 0:
                acc = epoch_acc(model, 
                                BATCH_SIZE, 
                                train_component,
                                embed_layer,dev_data,
                                table_type=TABLE_TYPE)
                if acc > best_acc:
                    best_acc = acc
                    print("Save model...")
                    torch.save(model.state_dict(), 
                                SAVED_MODELS_FOLDER+"/{}_models.dump".format(train_component))




if __name__ == '__main__':
    nlq = "What are the names of the departments that were founded after 1800?"
    db_name = "department_management"
    correct_query = "Select Name from department where Creation > 1800"

    N_word=300
    B_word=42
    LOAD_USED_W2I = False
    USE_SMALL=False

    # Loading Pretrained Word Embeddings
    word_emb = load_word_emb(file_name = 'glove/glove.%dB.%dd.txt'%(B_word,N_word), \
                                load_used=LOAD_USED_W2I, 
                                use_small=USE_SMALL)
    print("word_emb type = {}".format(type(word_emb)))
    print("random element from word_emb = {}".format(word_emb[random.choice(list(word_emb.keys()))]))
    print("finished load word embedding")
    #word_emb = load_concat_wemb('glove/glove.42B.300d.txt', "/data/projects/paraphrase/generation/para-nmt-50m/data/paragram_sl999_czeng.txt")


    # train the network with the correct query
    train_feedback(nlq = nlq,                                            # What is the minimum department budget? 
                    db_name = db_name,                                                          # department_management
                    correct_query = correct_query,      # SELECT 
                    toy = False,
                    word_emb = word_emb)
    
    # infer the same question to see whether the output changed
    infer_script(nlq = nlq,
                db_name = db_name,
                toy = False,
                word_emb = word_emb)