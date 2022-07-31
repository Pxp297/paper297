from torch import nn
from torch.nn import functional as F
from transformer import PositionalEncoder,Encoder,DotAttention
import torch
class LogRepModel(nn.Module):#Model for LogRep and all the variations used in ablation test
    def __init__(self,embedding_size,d_k,d_ff,num_layers,num_heads,dropout,device,padding_size,seq_len,exclude_keys,connect_way,data_keys):
        super().__init__()
        self.embedding_size=embedding_size
        self.d_k=d_k
        self.d_ff=d_ff
        self.num_heads=num_heads
        self.num_layers=num_layers
        self.padding_size=padding_size
        self.seq_len=seq_len
        self.dropout=dropout
        self.exclude_keys=exclude_keys
        self.connect_way=connect_way
        self.data_keys=data_keys

        self.num_classes=2

        self.attention1=DotAttention(d_model=self.padding_size,dropout=self.dropout)
        self.dense1=nn.Linear(self.padding_size,self.embedding_size)
        self.attention2=DotAttention(d_model=self.padding_size,dropout=self.dropout)
        self.dense2=nn.Linear(self.padding_size,self.embedding_size)

        self.pos_encoder=PositionalEncoder(d_model=self.embedding_size,seq_len=self.seq_len)
        self.encoder=Encoder(n_layers=self.num_layers,d_model=self.embedding_size,d_k=self.d_k,d_ff=self.d_ff,dropout=self.dropout,num_heads=self.num_heads)

        self.dropout1 = nn.Dropout(self.dropout)
        self.linear1 = nn.Linear(self.embedding_size, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(self.dropout)
        self.bn2 = nn.BatchNorm1d(32)
        self.linear2 = nn.Linear(32, self.num_classes)
        self.softm=nn.Softmax()
    
    def forward(self,features,enc_mask=None):
        def _get_feature(key):
            try:
                return key,features[key]
            except:
                return key,None
        inputs_all=tuple(map(_get_feature,self.data_keys))
    
        for key,val in inputs_all:
            if self.exclude_keys.count(key)==0 and val is None:
                print(features)
                raise RuntimeError('Feature keys not match model setting: {} key has no data but is not excluded from setting'.format(key))
        assert(len(inputs_all)<=3)
        if len(inputs_all)<3:
            inputs_all=list(inputs_all)+[(None,None)]*(3-len(inputs_all))
        in1,in2,in3=(term[1] for term in inputs_all)

        if self.connect_way=='attn':
            if in3 is None:
                attn_output1=in2
            elif in2 is None:
                attn_output1=in3
            else:
                
                attn_output1,_=self.attention1(in3,in2,in2)

                attn_output1=self.dense1(attn_output1+in3)
                attn_output1=F.relu(attn_output1)
            
            if in1 is None:
                attn_output2=attn_output1
                classifier_input=attn_output2
            else:
                if attn_output1 is None:
                    attn_output2=in1
                    classifier_input=attn_output2
                else:
                    attn_output2,_=self.attention2(attn_output1,in1,in1)
                    pos_enc_output=self.pos_encoder(in1)
                    classifier_input=attn_output2+pos_enc_output
        elif self.connect_way=="concat":
            if in3 is None:
                attn_output1=in2
            elif in2 is None:
                attn_output1=in3
            else:
                output1=torch.cat((in3,in2),dim=2)
            if in1 is None:
                output2=output1
                classifier_input=output2
            else:
                if output1 is None:
                    output2=in1
                    classifier_input=output2
                else:
                    output2=torch.cat((output1,in1),dim=2)
                    classifier_input=output2

        else:
            raise NotImplementedError('{} is not valid option'.format(self.connect_way))

        if classifier_input is None:
            raise RuntimeError("No data left for model, exclude keys:{}".format(self.exclude_keys))
        enc_output=self.encoder(classifier_input,enc_mask)
        enc_output=torch.mean(enc_output,1)
        enc_output=self.dropout1(enc_output)
        prediction=F.relu(self.linear1(enc_output))
        prediction=self.dropout2(prediction)
        prediction=self.linear2(prediction)

        return prediction