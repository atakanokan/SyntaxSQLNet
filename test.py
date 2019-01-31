"""
python test.py --toy --models saved_models --test_data_path data/spider/dev.json --output_path output_test.txt
"""


import json
import torch
import datetime
import argparse
import numpy as np
from utils import *
from supermodel import SuperModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_emb', action='store_true',
            help='Train word embedding.')
    parser.add_argument('--toy', action='store_true',
                        help='If set, use small data; used for fast debugging.')
    parser.add_argument('--models', type=str, help='path to saved model')
    parser.add_argument('--test_data_path',type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--history_type', type=str, default='full', 
                        choices=['full','part','no'], 
                        help='full, part, or no history')
    parser.add_argument('--table_type', type=str, default='std', 
                        choices=['std','hier','no'], 
                        help='standard, hierarchical, or no table info')
    parser.add_argument('--gpu_enable', default=False, action='store_true',
                        help='enable GPU')
    args = parser.parse_args()
    use_hs = True
    if args.history_type == "no":
        args.history_type = "full"
        use_hs = False

    N_word = 300
    B_word = 42
    N_h = 300
    N_depth = 2  # not used in test.py

    GPU_ENABLE = args.gpu_enable
    SAVED_MODELS_FOLDER = args.models
    
    # if args.part:
    #     part = True
    # else:
    #     part = False
    if args.toy:
        USE_SMALL=True
        # GPU=True
        GPU = args.gpu_enable
        BATCH_SIZE=1 # doesnt work with anything other than 1, if >1 just duplicates everything #default:20 
    else:
        USE_SMALL=False
        # GPU=True
        GPU = args.gpu_enable
        BATCH_SIZE=1 # doesnt work with anything other than 1, if >1 just duplicates everything #default:64 
    # TRAIN_ENTRY=(False, True, False)  # (AGG, SEL, COND)
    # TRAIN_AGG, TRAIN_SEL, TRAIN_COND = TRAIN_ENTRY
    learning_rate = 1e-4

    #TODO
    print("args.test_data_path = " + str(args.test_data_path))
    data = json.load(open(args.test_data_path))
    # dev_data = load_train_dev_dataset(args.train_component, "dev", args.history)

    word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word), \
            load_used=args.train_emb, 
            use_small=USE_SMALL)
    # dev_data = load_train_dev_dataset(args.train_component, "dev", args.history)
    #word_emb = load_concat_wemb('glove/glove.42B.300d.txt', "/data/projects/paraphrase/generation/para-nmt-50m/data/paragram_sl999_czeng.txt")

    print("table_type = " + str(args.table_type))
    model = SuperModel(word_emb, 
                        N_word=N_word, 
                        gpu=GPU, 
                        trainable_emb = args.train_emb, 
                        table_type=args.table_type, 
                        use_hs=use_hs)

    # agg_m, sel_m, cond_m = best_model_name(args)
    # torch.save(model.state_dict(), "saved_models/{}_models.dump".format(args.train_component))

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

    # TEST THE MODEL
    # Evaluation criteria: None, because the correct queries are on the Spider website
    # https://yale-lily.github.io/spider

    test_acc(model, BATCH_SIZE, data, args.output_path)


    #test_exec_acc()
