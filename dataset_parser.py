# Import models
import os
import string
import pandas as pd
import numpy as np
from collections import OrderedDict
import re
from tqdm import tqdm
from sklearn.utils import shuffle
import pickle
import random
import time
import torch
import sys

#pre-trained language models
from transformers import BertTokenizer,BertModel

#random seed control
random.seed(1234) 
np.random.seed(1234)

# os.environ['CUDA_VISIBLE_DEVICES']="1" # CUDA device setting

def bert_init(path):
    global global_sub_model,global_tokenizer
    global_tokenizer=BertTokenizer.from_pretrained(path)
    global_sub_model=BertModel.from_pretrained(path)
def bert_encode(s,no_wordpiece=0):
    global global_sub_model,global_tokenizer
    if no_wordpiece:
        words = s.split(" ")
        words = [word for word in words if word in global_tokenizer.vocab.keys()]
        s = " ".join(words)
    inputs = global_tokenizer(s, return_tensors='pt', padding=True)

    outputs=global_sub_model(**inputs)
    v = torch.mean(outputs.last_hidden_state, 1)
    return v[0]

def str_parse(s,padding_size):#split string into structured parts
        
    def _split_str_num(s):
        s = s.split()
        str_result = []
        num_result = []
        positoin_result = []
        for index, ele in enumerate(s):
            num_patterns = re.findall('\d', ele)
            if num_patterns:
                for num_pattern in num_patterns:
                    num_result.append(int(num_pattern))
                    positoin_result.append(index + 1)
            else:
                str_result.append(ele)

        str_result = " ".join(str_result)     
        if len(num_result) > padding_size:
            print("The length is {} >= padding size".format(len(num_result)))
            return str_result, num_result, positoin_result

        assert len(num_result) <= padding_size
        for _ in range(padding_size - len(num_result)):
            num_result.append(-1)
            positoin_result.append(0)
        return str_result, num_result, positoin_result
    s = re.sub('\]|\[|\)|\(|\=|\,|\;', ' ', s)
    s = " ".join([word.lower() if word.isupper() else word for word in s.strip().split()])
    s = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', s))
    trantab = str.maketrans(dict.fromkeys(list(string.punctuation), " "))
    content = s.translate(trantab)
    s = " ".join([word.lower().strip() for word in content.strip().split()])
    return _split_str_num(s)

def load_HDFS(log_file, label_file,
              tokenizer_path,
              save_pickle,save_csv=None,
              e_type="bert",
              padding_size=64,
              parsing_length=None,
              event_key='EventSequence',
              num_key='NumSequence',
              pos_key='PositionSequence',
              label_key='Label'):


    print('====== Input data start ======')
    print("extract data from {}, use label file {}\nuse tokenizer at {},save file {}, use data keys:{}".format(log_file,label_file,tokenizer_path,save_pickle,(event_key,num_key,pos_key)))

    encoding_dict = {}
    if e_type == "bert":
        bert_init(tokenizer_path)
        encoder=bert_encode

    encoding_dict = {}
    t0 = time.time()
    

    if log_file.endswith('.log'):
        
        print("Loading", log_file)
        with open(log_file, mode="r", encoding='utf8') as f:
            logs = f.readlines()
            logs = [x.strip() for x in logs]
        print("Data Loaded")
        event_dict = OrderedDict()
        num_dict = OrderedDict()
        pos_dict = OrderedDict()
        time_dict = OrderedDict()

        n_logs = len(logs)
        print("total logs:{}".format(n_logs))
        
        max_num = 0
        for i, line in enumerate(logs):
            blkId_list = re.findall(r'(blk_-?\d+)', line)

            line = re.sub(r'(blk_-?\d+)', ' ', line)
            
            blkId_list = list(set(blkId_list))
            if len(blkId_list) >= 2:
                continue
            blkId_set = set(blkId_list)

            time_str = line.split(" ")[0] + line.split(" ")[1]

            content, num_vector, position_vector = str_parse(line.lower(),padding_size)
            if len(num_vector) > max_num:
                max_num = len(num_vector)

            if content not in encoding_dict.keys():
                encoding_dict[content] = encoder(content, 0)
                
            for blk_Id in blkId_set:
                if not blk_Id in event_dict:
                    event_dict[blk_Id] = []
                    num_dict[blk_Id] = []
                    pos_dict[blk_Id] = []
                    time_dict[blk_Id] = []
                event_dict[blk_Id].append(encoding_dict[content])
                num_dict[blk_Id].append(np.array(num_vector).astype("float"))
                pos_dict[blk_Id].append(np.array(position_vector).astype("float"))
                time_dict[blk_Id] = time_str

            i += 1
            
            if i % 1000 == 0 or i == n_logs:
                print("Loading {0:.2f} \%- number of unique message: {1}".format(i / n_logs * 100, len(encoding_dict.keys())),flush=True)
                print(parsing_length,i,flush=True)
            if parsing_length is not None:
                if i>=parsing_length:
                    print("reached target limit {}".format(parsing_length))
                    break
        print("max_length is ", max_num)


        event_data_frame = pd.DataFrame(list(event_dict.items()), columns=['BlockId', event_key])
        num_data_frame = pd.DataFrame(list(num_dict.items()), columns=['BlockId', num_key])
        pos_data_frame = pd.DataFrame(list(pos_dict.items()), columns=['BlockId', pos_key])
        time_df = pd.DataFrame(list(time_dict.items()), columns=['BlockId', 'Time'])
        time_df['Time'] = pd.to_datetime(time_df['Time'],format='%y%m%d%H%M%S')

        full_data_frame = pd.concat([event_data_frame, num_data_frame[num_key], pos_data_frame[pos_key],time_df['Time']], axis = 1)


        
        # Split training and validation set in a class-uniform way
        label_data = pd.read_csv(label_file, engine='c', na_filter=False, memory_map=True)
        label_data = label_data.set_index('BlockId')
        label_dict = label_data[label_key].to_dict()
        full_data_frame[label_key] = full_data_frame['BlockId'].apply(lambda x: 1 if label_dict[x] == 'Anomaly' else 0)
        
        print("Saving data...")
        
        if save_pickle is not None:
            full_data_frame.to_pickle(save_pickle)

        if save_csv is not None:
            full_data_frame.to_csv('data_instances.csv', index=False)
        print("finished at {}".format(time.time() - t0))
        return full_data_frame
    else:
        raise NotImplementedError('Not supported data type')

def load_BGL(log_file, window_size,tokenizer_path,save_pickle,padding_size=200,parsing_length=None,e_type='bert'):
    if e_type == "bert":
        bert_init(tokenizer_path)
        encoder=bert_encode
    else:
        raise NotImplementedError
    
    if log_file.endswith('.log'):
        print("Loading", log_file)
        with open(log_file, mode="r") as f:
            logs = f.readlines()
            logs = [x.strip() for x in logs]
        E={}
        x_total, y_total = [], []
        x_total_num, x_total_position = [], []
        n_total = len(logs)
        t0 = time.time()
        anomaly_count=0

        c = 0 #log item count
        i = 0 #window start index
        while i < n_total - window_size:
            c += 1
            if c % 1000 == 0:
                print("Loading {0:.2f} - {1} unique logs".format(i * 100 / n_total, len(E.keys())),flush=True)
                print(c,parsing_length)
            if parsing_length is not None and parsing_length<=c:
                print("Parsing stopped at the {}th log".format(c))
                break
            if logs[i][0] != "-":
                anomaly_count += 1
            seq = []
            seq_by = []
            seq_bert = []
            num_seq = []
            position_seq = []
            max_num = 0
            label = 0
            # Non-overlapping
            for j in range(i, i + window_size):
                if logs[j][0] != "-":
                    label = 1
                content = logs[j]

                content = content[content.find(' ') + 1:]
                
                content = " ".join(content.split(" ")[7:])

                original_content = content.lower()
                content, num_vector, position_vector = str_parse(content.lower(),padding_size=padding_size)

                assert len(num_vector) == padding_size
                if len(num_vector) > max_num:
                    max_num = len(num_vector)

                if content not in E.keys():
                    encoder(content, 0)
                    try:
                        E[content] = encoder(content, 0)
                    except:
                        raise RuntimeError('Encoding {} failed'.format(content) )
                emb = E[content]
                seq.append(emb)

                num_seq.append(np.array(num_vector).astype("float"))
                position_seq.append(np.array(position_vector).astype("float"))
            x_total.append(seq.copy())
            x_total_num.append(num_seq.copy())
            x_total_position.append(position_seq.copy())
            y_total.append(label)
            i = i + window_size
        print("max_num = ", max_num)
        print("last train index:", i)

        print(time.time() - t0)
        data_df = pd.DataFrame({'EventSequence':x_total,
                            'NumSequence': x_total_num,'PositionSequence': x_total_position, 'Label':y_total})
        if save_pickle is not None:
            print("saving data to {}".format(save_pickle))
            data_df.to_pickle(save_pickle)
        return data_df
