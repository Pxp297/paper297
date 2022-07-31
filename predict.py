
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm
import numpy as np
class Predicter():
    def __init__(self, model,checkpoint_path, test_data,device,batch_size):
        self.test_loader=DataLoader(test_data,batch_size=batch_size,shuffle=False ,pin_memory=True)
        self.model=model
        self.device=device
        self.model_path=checkpoint_path

    def predict(self):
        model=self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        print("use model path: {}".format(self.model_path))
        print("test length :{}".format(len(self.test_loader)))

        data_body=tqdm(self.test_loader)
        predictions=[]
        groudtruths=[]
        full_len=len(self.test_loader)
        for i, (data,label) in enumerate(data_body):
            with torch.no_grad():
                features={}
                for key,value in data.items():
                    features[key]=value.clone().detach().to(self.device)
                output=model(features=features)
                prediction=np.argmax(output.cpu(),axis=1)
                predictions.append(prediction)
                label=label.reshape([label.shape[0]])
                groudtruths.append(label.cpu())
        predictions=torch.cat(predictions)
        groudtruths=torch.cat(groudtruths)
        report=classification_report(groudtruths.numpy(),predictions.numpy(),digits=5)

        return report