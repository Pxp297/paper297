# LogRep
This is a basic implementation for testing the representation method LogRep for log based anomaly detection. 

# Environment & Dependencies
This project is based on the pytorch deep learning tools and the pandas library for data processing. The envrioment required for executing the program is listed below:

```
python                    3.8.0 

torch                     1.11.0 
pyparsing                 3.0.8 
numpy                     1.22.3
cudnn                     8.2.1.32
cudatoolkit               11.1.1
scikit-learn              1.0.2
scipy                     1.8.0 
tqdm                      4.64.0 
transformers              4.18.0 
```

The pre-trained Bert models can be downloaded from https://huggingface.co/bert-base-uncased

# Datasets
We used two public datasets for log analysis, HDFS and BGL. In the "dataset" directory there are two sample log files. The complete version of the two datasets can be fetched at https://github.com/logpai/loghub. 

# Usage
Use `"python main.py [cmd]"` to run `LogRep`, `[cmd]` option can be `parse` or `test`. Use `parse` to parse raw logs. Use `test` to train and test anomaly detection model with the data generated in the parsing stage.

To run `LogRep` to process the log file and generate the pandas DataFrame for representation generation, use `"python main.py parse"`, with `config.json` set to suitable parameters like:
```
{
    "parse":{
        "dataset":"hdfs",
        "parameters":{
            "log_file":"./dataset/HDFS.log",
            "e_type":"bert",
            "save_pickle":"hdfs_rep.gzip",
            "tokenizer_path":"./language_models/bert-base-uncased",
            "parsing_length":null
        }
        
    }
}
```
In the case above, the result DataFrame will be saved in "hdfs_rep.gzip". Then the data file can be utilized for attention based representation generation and anomaly detection test. 

To run `LogRep` to do anomaly detection, use `"python main.py test"` with `config.json` like:
```
{
"train":{
        "device":"cuda",
        "model_name":"LogRep_Vanilla",
        "save_step":5,
        "save_path":"./model_save",
        "batch_size":64,
        "max_epoch":40,
        "init_lr":0.001
    },
    "model":{
        "embedding_size":768,
        "d_k":768,
        "d_ff":2048,
        "num_layers":1,
        "num_heads":10,
        "dropout":0.1,
        "padding_size":64,
        "device":"cuda",
        "seq_len":50,
        "exclude_keys":[],
        "connect_way":"attn",
        "data_keys":["EventSequence","NumSequence","PositionSequence"]
    },
    "representation-data":["./hdfs_data.gzip"],
    "train_set_length":8000,
    "random_offset":true
}
```
## Parameter setting
The `train_set_length` can be set to various values to examine the performance of models with different training data length, and the `random_offset` is turned on to produce multiple random results with different proportion of the data used as training set.

The `exclude_keys` can be used to exclude part of the representation to compare the contribution of each part in the performance of the anomaly detection model. The `connect_way` can be `"attn"` or `"concat"`. `"attn"` means the representation method will be based on attention layer, and `"concat"` will set the representation with pure contenation way. Use `"concat"` and set `"exclude_keys"` to `["PositionSequence","NumSequence"]` to test the performance of `LogRep_S`, `LogRep_SN`, `LogRep_SPN` and `LogRep_SP`. The related parameters should be set as follows:
```
HDFS dataset
Variation   embedding_size/d_k  paddings_size
LogRep_SPN     896              64
LogRep_SP      832              64
LogRep_SN      832              64
LogRep_S       768              64
LogRep         768              64

BGL dataset
LogRep_SPN     1168             64
LogRep_SP      968              64
LogRep_SN      968              64
LogRep_S       768              64
LogRep         768              64
```

With the `random_offset` option turned on 
