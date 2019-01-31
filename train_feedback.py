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

TRAIN_COMPONENTS = ('multi_sql','keyword','col','op','agg','root_tem','des_asc','having','andor')
SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND', 'EQL', 'GT', 'LT', '<BEG>']

def train_feedback(nlq,db_name,correct_query,toy):
    """
    Arguments:
        nlq: english question (tokenization is done here) - get from Flask (User)
        db_name: name of the database the query targets - get from Flask (User)
        correct_query: the ground truth query supplied by the user(s) - get from Flask
        toy: uses a small example of word embeddings to debug faster
    """

    EPOCH = 1

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

    # Loading Pretrained Word Embeddings
    word_emb = load_word_emb(file_name = 'glove/glove.%dB.%dd.txt'%(B_word,N_word), \
                                load_used=TRAIN_EMB, 
                                use_small=USE_SMALL)
    print("word_emb type = {}".format(type(word_emb)))
    print("random element from word_emb = {}".format(word_emb[random.choice(list(word_emb.keys()))]))
    print("finished load word embedding")
    #word_emb = load_concat_wemb('glove/glove.42B.300d.txt', "/data/projects/paraphrase/generation/para-nmt-50m/data/paragram_sl999_czeng.txt")
    

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
        print("train_data type: {}".format(type(train_data)))
        # dev_data = load_train_dev_dataset(args.train_component, "dev", 
        #                                     args.history_type, 
        #                                     args.data_root)
        # sql_data, table_data, val_sql_data, val_table_data, \
        #         test_sql_data, test_table_data, \
        #         TRAIN_DB, DEV_DB, TEST_DB = load_dataset(args.dataset, use_small=USE_SMALL)


        
        # Selecting which Model to Train
        model = None
        if train_component == "multi_sql":
            model = MultiSqlPredictor(N_word=N_word,  
                                    N_h=N_h,
                                    N_depth=N_depth,
                                    gpu=GPU, 
                                    use_hs=use_hs)
        elif train_component == "keyword":
            model = KeyWordPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,
                                    gpu=GPU, use_hs=use_hs)
        elif train_component == "col":
            model = ColPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,
                                gpu=GPU, use_hs=use_hs)
        elif train_component == "op":
            model = OpPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,
                                gpu=GPU, use_hs=use_hs)
        elif train_component == "agg":
            model = AggPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,
                                gpu=GPU, use_hs=use_hs)
        elif train_component == "root_tem":
            model = RootTeminalPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,
                                            gpu=GPU, use_hs=use_hs)
        elif train_component == "des_asc":
            model = DesAscLimitPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,
                                        gpu=GPU, use_hs=use_hs)
        elif train_component == "having":
            model = HavingPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,
                                    gpu=GPU, use_hs=use_hs)
        elif train_component == "andor":
            model = AndOrPredictor(N_word=N_word, N_h=N_h, N_depth=N_depth, 
                                    gpu=GPU, use_hs=use_hs)
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
        for i in range(EPOCH):
            print('Epoch %d @ %s'%(i+1, datetime.datetime.now()))
            # arguments of epoch_train
            # model, optimizer, batch_size, component,embed_layer,data, table_type
            # print(' Loss = %s' % epoch_train(
            #                     model, optimizer, BATCH_SIZE,
            #                     args.train_component,
            #                     embed_layer,
            #                     train_data,
            #                     table_type=args.table_type))
            # acc = epoch_acc(model, 
            #                 BATCH_SIZE, 
            #                 args.train_component,
            #                 embed_layer,dev_data,
            #                 table_type=args.table_type)
            # if acc > best_acc:
            #     best_acc = acc
            #     print("Save model...")
            #     torch.save(model.state_dict(), 
            #                 args.save_dir+"/{}_models.dump".format(args.train_component))
            print(' Loss = %s' % epoch_feedback_train(model = model, 
                                                    optimizer = optimizer, 
                                                    batch_size = BATCH_SIZE, 
                                                    component = train_component, 
                                                    embed_layer = embed_layer, 
                                                    data = train_data, 
                                                    table_type = TABLE_TYPE,
                                                    nlq = nlq,
                                                    db_name = db_name,
                                                    correct_query = correct_query))




if __name__ == '__main__':
    # train the network with the correct query
    train_feedback(nlq = "Which department has the minimum budget?",                                            # What is the minimum department budget? 
                    db_name = "department_management",                                                          # department_management
                    correct_query = "select Name from department order by Budget_in_Billions asc limit 1",      # SELECT 
                    toy = True)
    
    # infer the same question to see whether the output changed
