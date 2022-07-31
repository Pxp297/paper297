
class Dataset:
    def __init__(self,src_data):
        self.src_data=src_data
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import random
class LabeledFramesSet(Dataset):
    def __init__(self, src_data,x_column_keys,y_column_key):
        super().__init__(src_data)
        
        if src_data is not None:
            
            self.x_frames=self.src_data[x_column_keys]
            self.y_frames=self.src_data[y_column_key]
        self.x_keys=x_column_keys
        self.y_key=y_column_key
        self.src_data=None
        self.length=len(self.y_frames) if src_data is not None else 0
    def Oversample(self,type_y_value=1,ratio=None):

        target_index=self.y_frames==type_y_value
        target_data=self.x_frames[target_index]
        base_length=len(target_data)
        
        if ratio is None:
            ratio=(self.length-base_length)/base_length
            print("without ratio definition, use {} as ratio".format(ratio))
        concat_data=target_data
        while ratio>0:
            target_data=pd.concat([target_data,concat_data],axis=0,ignore_index=True)
            ratio-=1
        
        growth_length=len(target_data)
        print("growth length: {}".format(growth_length))

        self.x_frames=pd.concat([self.x_frames,target_data],axis=0,ignore_index=True)
        self.y_frames=pd.concat([self.y_frames,pd.DataFrame([type_y_value]*growth_length)],axis=0,ignore_index=True)
        
    def SplitSet(self,split_length,random_offset=True):
        new_set=self.copy_main()
        if random_offset:
            offset=random.randint(split_length,self.length-split_length)
            new_set.x_frames=pd.concat([self.x_frames[:offset],self.x_frames[offset+split_length+1:]],axis=0,ignore_index=True)
            new_set.y_frames=pd.concat([self.y_frames[:offset],self.y_frames[offset+split_length+1:]],axis=0,ignore_index=True)
            new_set.length=len(new_set.x_frames)
            print("using data at [{},{}) as training set".format(offset,offset+split_length))
        else:
            new_set.x_frames=self.x_frames[split_length+1:]
            new_set.y_frames=self.y_frames[split_length+1:]
            new_set.length=len(new_set.x_frames)
            offset=0
        self.x_frames=self.x_frames[offset:split_length+offset]
        self.y_frames=self.y_frames[offset:split_length+offset]
        self.length=len(self.y_frames)
        return new_set
    def copy_main(self):
        Construct=type(self)
        new_set=Construct(None,x_column_keys=self.x_keys,y_column_key=self.y_key)
        return new_set
    def __str__(self):
        return "x_frames:\nkeys:{}\nframe_body:{}\ny_frames:\nkeys:{}\nframe:{}\n".format(self.x_keys,self.x_frames,self.y_key,self.y_frames)
    def __len__(self):
        return self.length
class TorchDatasetLabeledFrames(LabeledFramesSet):
    def __init__(self, src_data, x_column_keys, y_column_key,max_length,stack_columns):
        super().__init__(src_data, x_column_keys, y_column_key)
        self.max_length=max_length
        self.transform_flag=False
        self.stack_columns=stack_columns
        
        for k in self.stack_columns:
            if self.x_keys.count(k) !=0:
                self.x_keys.remove(k)
    def prepare_train(self):
        print("status of prepare {}".format("refresh" if self.transform_flag else "startup"))
        self.transform_flag=True
        self.np_values_for_tensors={data_key:np.array(self.x_frames[data_key].values) for data_key in self.x_keys}
        self.np_value_for_stack={data_key:self.x_frames[data_key].values for data_key in self.stack_columns}
        #self.np_value_y=self.y_frames[self.y_key].values
        self.np_value_y=np.array(self.y_frames.values)
    def __getitem__(self,idx):
        if not self.transform_flag:
            self.prepare_train()
        torch_data_dict={data_key:torch.stack(self.np_value_for_stack[data_key][idx]) for data_key in self.stack_columns}
        for data_key in self.x_keys:
            torch_data_dict[data_key]=torch.tensor(self.np_values_for_tensors[data_key][idx],dtype=torch.float)
        length=max([len(l) for l in list(torch_data_dict.values())])
        if length>self.max_length:
            for k in torch_data_dict:
                val=torch_data_dict[k][-self.max_length:]
                torch_data_dict[k]=val
                length=max([len(l) for l in list(torch_data_dict.values())])
        
        extend_filling=(0,0,0,self.max_length-length)
        
        for k in torch_data_dict:
            torch_data_dict[k]=F.pad(torch_data_dict[k],extend_filling,'constant',value=0)
        
        return torch_data_dict,self.np_value_y[idx]
    
    def SplitSet(self, split_length,random_offset=True):
        self.transform_flag=False
        self.np_value_for_stack=None
        self.np_value_y=None
        self.np_values_for_tensors=None
        new_set=super().SplitSet(split_length,random_offset)
        
        new_set.max_length=self.max_length
        new_set.stack_columns=self.stack_columns
        return new_set
    def copy_main(self):
        return TorchDatasetLabeledFrames(None,self.x_keys,self.y_key,self.max_length,self.stack_columns)
    def __str__(self):
        return "x_frames:\nkeys:{} stack_keys:{}\nframe_body:{}\ny_frames:\nkeys:{}\nframe:{}\n".format(self.x_keys,self.stack_columns,self.x_frames,self.y_key,self.y_frames)
