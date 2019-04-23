This is a Python 3.6.1 and PyTorch 1.0 implementation of the paper referenced below.


## SyntaxSQLNet: Syntax Tree Networks for Complex and Cross-DomainText-to-SQL Task

Source code of our EMNLP 2018 paper: [SyntaxSQLNet: Syntax Tree Networks for Complex and Cross-DomainText-to-SQL Task
](https://arxiv.org/abs/1810.05237).

### Citation

```
@InProceedings{Yu&al.18.emnlp.syntax,
  author =  {Tao Yu and Michihiro Yasunaga and Kai Yang and Rui Zhang and Dongxu Wang and Zifan Li and Dragomir Radev},
  title =   {SyntaxSQLNet: Syntax Tree Networks for Complex and Cross-Domain Text-to-SQL Task},
  year =    {2018},  
  booktitle =   {Proceedings of EMNLP},  
  publisher =   {Association for Computational Linguistics},
}
```

#### Presentation on the Business Use Case
Please look at Atakan_Okan_Text2SQL.pdf in main directory.


#### Environment Setup

1. The code uses Python 3.7 and [Pytorch 1.0.0](https://pytorch.org/previous-versions/) GPU.
2. Install Python dependency: `pip install -r requirements.txt`

#### Download Data, Embeddings, Scripts, and Pretrained Models
1. Download the dataset from [the Spider task website](https://yale-lily.github.io/spider) to be updated, and put `tables.json`, `train.json`, and `dev.json` under `data/` directory.
2. Download the pretrained [Glove](https://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip), and put it as `glove/glove.%dB.%dd.txt`
3. Download `evaluation.py` and `process_sql.py` from [the Spider github page](https://github.com/taoyds/spider)
4. Download preprocessed train/dev datasets and pretrained models from [here](https://drive.google.com/file/d/1FHEcceYuf__PLhtD5QzJvexM7SNGnoBu/view?usp=sharing). It contains: 
   -`generated_datasets/`
    - ``generated_data`` for original Spider training datasets, pretrained models can be found at `generated_data/saved_models`
    - ``generated_data_augment`` for original Spider + augmented training datasets, pretrained models can be found at `generated_data_augment/saved_models`

#### Generating Train/dev Data for Modules
You could find preprocessed train/dev data in ``generated_datasets/``.

To generate them by yourself, update dirs under `TODO` in `preprocess_train_dev_data.py`, and run the following command to generate training files for each module:
```
python preprocess_train_dev_data.py train|dev
```

#### Folder/File Description
- ``data/`` contains raw train/dev/test data and table file
- ``generated_datasets/`` described as above
- ``models/`` contains the code for each module.
- ``evaluation.py`` is for evaluation. It uses ``process_sql.py``.
- ``train.py`` is the main file for training. Use ``train_all.sh`` to train all the modules (see below).
- ``test.py`` is the main file for testing. It uses ``supermodel.sh`` to call the trained modules and generate SQL queries. In practice, and use ``test_gen.sh`` to generate SQL queries.
- `generate_wikisql_augment.py` for cross-domain data augmentation


#### Training
Run ``train_all.sh`` to train all the modules.
It looks like:
```
python train.py \
    --data_root       path/to/generated_data \
    --save_dir        path/to/save/trained/module \
    --history_type    full|no \
    --table_type      std|no \
    --train_component <module_name> \
    --epoch           <num_of_epochs>
```

#### Testing
Run ``test_gen.sh`` to generate SQL queries.
``test_gen.sh`` looks like:
```
SAVE_PATH=generated_datasets/generated_data/saved_models_hs=full_tbl=std
python test.py \
    --test_data_path  path/to/raw/test/data \
    --models          path/to/trained/module \
    --output_path     path/to/print/generated/SQL \
    --history_type    full|no \
    --table_type      std|no \
```

#### Flask Testing (Local)
Run model with question = `` What are the maximum and minimum budget of the departments?`` and database name = `` department_management ``

#### Docker image creation and push to Docker Hub
* ` docker build -t model-app `
* ` docker login` -> enter your credentials
* ` docker images ` -> get the image id of the model's container
* ` docker tag <your image id> <your docker hub id>/<app name> `
* ` docker push <your docker hub name>/<app-name> `

#### Kubernetes Deployment
After pushing the Docker image to Docker Hub & creating the Kubernetes cluster; run the following in Cloud Shell:
* ` kubectl run model-app --image=atakanokan/model-app --port 5000 `
* Verify by ` kubectl get pods `
* `kubectl expose deployment model-app --type=LoadBalancer --port 80 --target-port 5000`
* `kubectl get service` and get the cluster-ip

And run the following from local terminal:
* ` curl -X GET 'http://<your service IP>/output?english_question=What+are+the+maximum+and+minimum+budget+of+the+departments%3F&database_name=department_management' `

#### Evaluation
Follow the general evaluation process in [the Spider github page](https://github.com/taoyds/spider).

#### Cross-Domain Data Augmentation
You could find preprocessed augmented data at `generated_datasets/generated_data_augment`. 

If you would like to run data augmentation by yourself, first download `wikisql_tables.json` and `train_patterns.json` from [here](https://drive.google.com/file/d/1DZCITYsy9oXjayY1I2e4Wk1nfxUa_oiG/view?usp=sharing), and then run ```python generate_wikisql_augment.py``` to generate more training data.

#### Acknowledgement

The implementation is based on [SQLNet](https://github.com/xiaojunxu/SQLNet). Please cite it too if you use this code.
