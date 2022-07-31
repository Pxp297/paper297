from train import Trainer
from predict import Predicter
from dataset import TorchDatasetLabeledFrames as Dataset
from model import LogRepModel
from dataset_parser import load_HDFS,load_BGL
import pandas
import json
import sys
options={}
with open("config.json",'r')as config_file:
    options=json.loads(config_file.read())

def parse():
    if options['parse']['dataset']=='hdfs':
        load_HDFS(**options['parse']['parameters'])
    elif options['parse']['dataset']=='bgl':
        load_BGL(**options['parse']['parameters'])
    else:
        raise NotImplementedError("Not supported dataset format {}".format(options['parse']['dataset']))

def test():
    rep_data_path=options['representation-data']
    if len(rep_data_path)==1:
        rep_data=pandas.read_pickle(rep_data_path[0])
        full_set=Dataset(src_data=rep_data,x_column_keys=["EventSequence","NumSequence","PositionSequence"],max_length=50,y_column_key="Label",stack_columns=["EventSequence"])
        train_len=options['train_set_length']
        train_set=full_set
        test_set=train_set.SplitSet(train_len)
    elif len(rep_data_path)==2:
        train_data_path=rep_data_path[0]
        test_data_path=rep_data_path[1]
        train_data=pandas.read_pickle(train_data)
        test_data=pandas.read_pickle(test_data)
        train_set=Dataset(src_data=train_data,x_column_keys=["EventSequence","NumSequence","PositionSequence"],max_length=50,y_column_key="Label",stack_columns=["EventSequence"])
        test_set=Dataset(src_data=test_data,x_column_keys=["EventSequence","NumSequence","PositionSequence"],max_length=50,y_column_key="Label",stack_columns=["EventSequence"])
    else:
        raise NotImplementedError("The representation data path parameter should be a single file or two files")
    train_set.Oversample()

    test_model=LogRepModel(**options['model'])

    train_config=options['train']
    train_config['model']=test_model
    train_config['data']=train_set
    trainer=Trainer(**train_config)
    trainer.start_train()
    predicter=Predicter(model=test_model,checkpoint_path=trainer.last_savepath,test_data=test_set,device=trainer.device,batch_size=trainer.batch_size)
    report=predicter.predict()
    print(report)

arg_exec=sys.argv[1]
if arg_exec=="parse":
    parse()
elif arg_exec=="test":
    test()
else:
    print("Cannot use option {}".format(arg_exec))