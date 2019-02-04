"""
This file will be used at inference time when the user clicks the 
'GENERATE SQL' button on the website

python inference.py --toy --models saved_models --output_path output_inference.txt


                                --test_data_path data/spider/dev.json 
"""


import json
import torch
import datetime
import argparse
import numpy as np
from utils import *
from supermodel import SuperModel
from data.process_sql import tokenize
import pickle
import os 



def infer_script(nlq,db_name,toy,word_emb):
    """
    Arguments:
        nlq: english question (tokenization is done here)
        db_name: name of the database the query targets
        toy: uses a small example of word embeddings to debug faster

    """

    SAVED_MODELS_FOLDER = "saved_models"
    OUTPUT_PATH = "output_inference.txt"
    HISTORY_TYPE = "full"
    GPU_ENABLE = False
    TRAIN_EMB = False
    TABLE_TYPE = "std"
    LOAD_USED_W2I = False

    use_hs = True
    if HISTORY_TYPE == "no":
        HISTORY_TYPE = "full"
        use_hs = False

    N_word=300
    B_word=42
    N_h = 300
    N_depth=2  # not used in test.py
    
    if toy:
        USE_SMALL=True
    else:
        USE_SMALL=False

    GPU = GPU_ENABLE
    BATCH_SIZE=1 #64


    # QUESTION TOKENIZATION
    tok_q = tokenize(nlq)
    # print("tokenized question: {}".format(tokenize("What are the maximum and minimum budget of the departments?")))

    # Natural language question and database reading
    nlq = {'db_id': db_name, 
            'question_toks': tok_q}
    print("nlq: {}".format(nlq))

    db_id = nlq["db_id"]

    table_dict = get_table_dict("./data/spider/tables.json")
    # table_dict = table_dict[db_id] # subset table dict to the specified database
    # table_dict[db_id] = {'column_names': [[-1, '*'], [0, 'department id'], [0, 'name'], [0, 'creation'], [0, 'ranking'], [0, 'budget in billions'], [0, 'num employees'], [1, 'head id'], [1, 'name'], [1, 'born state'], [1, 'age'], [2, 'department id'], [2, 'head id'], [2, 'temporary acting']], 
    #         'column_names_original': [[-1, '*'], [0, 'Department_ID'], [0, 'Name'], [0, 'Creation'], [0, 'Ranking'], [0, 'Budget_in_Billions'], [0, 'Num_Employees'], [1, 'head_ID'], [1, 'name'], [1, 'born_state'], [1, 'age'], [2, 'department_ID'], [2, 'head_ID'], [2, 'temporary_acting']], 
    #         'column_types': ['text', 'number', 'text', 'text', 'number', 'number', 'number', 'number', 'text', 'text', 'number', 'number', 'number', 'text'], 
    #         'db_id': 'department_management', 
    #         'foreign_keys': [[12, 7], [11, 1]], 
    #         'primary_keys': [1, 7, 11], 
    #         'table_names': ['department', 'head', 'management'], 
    #         'table_names_original': ['department', 'head', 'management']}


    # LOAD WORD EMBEDDINGS
    # if not os.path.isfile('./glove/usedwordemb.pickle'):
    # print("Creating word embedding dictionary...")
    # word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word), \
    #         load_used=LOAD_USED_W2I, 
    #         use_small=USE_SMALL)
        # print("word_emb: {}".format(word_emb))
        # print("tyep word_emb: {}".format(type(word_emb)))
    #     with open('./glove/usedwordemb.pickle', 'wb') as handle:
    #         print("Saving word embedding as pickle...")
    #         pickle.dump(word_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # else:
    #     with open('./glove/usedwordemb.pickle', 'rb') as handle:
    #         print("Loading word embedding pickle...")
    #         word_emb = pickle.load(handle)
    

    # dev_data = load_train_dev_dataset(args.train_component, "dev", args.history)
    #word_emb = load_concat_wemb('glove/glove.42B.300d.txt', "/data/projects/paraphrase/generation/para-nmt-50m/data/paragram_sl999_czeng.txt")

    # print("table_type = " + str(args.table_type))
    model = SuperModel(word_emb, 
                        N_word=N_word, 
                        gpu=GPU, 
                        trainable_emb = TRAIN_EMB, 
                        table_type = TABLE_TYPE, 
                        use_hs=use_hs)

    # agg_m, sel_m, cond_m = best_model_name(args)
    # torch.save(model.state_dict(), "saved_models/{}_models.dump".format(args.train_component))

    print("Loading modules...")
    if GPU_ENABLE:
        map_to = "gpu"
    else:
        map_to = "cpu"

    # LOAD THE TRAINED MODELS
    model.multi_sql.load_state_dict(torch.load("{}/multi_sql_models.dump".format(SAVED_MODELS_FOLDER),map_location=map_to))
    model.key_word.load_state_dict(torch.load("{}/keyword_models.dump".format(SAVED_MODELS_FOLDER),map_location=map_to))
    model.col.load_state_dict(torch.load("{}/col_models.dump".format(SAVED_MODELS_FOLDER),map_location=map_to))
    model.op.load_state_dict(torch.load("{}/op_models.dump".format(SAVED_MODELS_FOLDER),map_location=map_to))
    model.agg.load_state_dict(torch.load("{}/agg_models.dump".format(SAVED_MODELS_FOLDER),map_location=map_to))
    model.root_teminal.load_state_dict(torch.load("{}/root_tem_models.dump".format(SAVED_MODELS_FOLDER),map_location=map_to))
    model.des_asc.load_state_dict(torch.load("{}/des_asc_models.dump".format(SAVED_MODELS_FOLDER),map_location=map_to))
    model.having.load_state_dict(torch.load("{}/having_models.dump".format(SAVED_MODELS_FOLDER),map_location=map_to))
    model.andor.load_state_dict(torch.load("{}/andor_models.dump".format(SAVED_MODELS_FOLDER),map_location=map_to))


    # print("Model used:")
    # print(model)

    # This should return the generated SQL query
    # test_acc(model, BATCH_SIZE, data, args.output_path)
    #test_exec_acc()

    # This should return the generated SQL query
    gen_sql = infer_sql(model = model, 
                        batch_size = BATCH_SIZE, 
                        nlq = nlq, 
                        table_dict = table_dict,
                        output_path = OUTPUT_PATH)
    # print("Generated SQL: {}".format(gen_sql))

    return gen_sql



if __name__ == '__main__':
    N_word=300
    B_word=42
    LOAD_USED_W2I = False
    USE_SMALL=True

    print("Creating word embedding dictionary...")
    word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word), \
            load_used=LOAD_USED_W2I, 
            use_small=USE_SMALL)


    infer_script(nlq = "Names of departments created after 1800",
                db_name = "department_management",
                toy = True,
                word_emb = word_emb)