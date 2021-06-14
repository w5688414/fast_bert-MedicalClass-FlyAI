# -*- coding: utf-8 -*
from flyai.framework import FlyAI
from path import MODEL_PATH, DATA_PATH

from fast_bert.prediction import BertClassificationPredictor
import pandas as pd

model_path = MODEL_PATH+'/model_out'
label_path = "./data/"


class Prediction(FlyAI):
    def load_model(self):
        self.predictor = BertClassificationPredictor(
            model_path=model_path,
            label_path=label_path,  # location for labels.csv file
            multi_label=False,
            model_type='bert',
            do_lower_case=False)
        

    def predict(self,text):
        single_prediction = self.predictor.predict_batch(text)
        # print(single_prediction)
        # print(len(single_prediction))
        return single_prediction

    # def predict(self, title, text):
    #     data = title + text
    #     if len(data) > 300:
    #         data = data[:300]
    #     single_prediction = self.predictor.predict(data)
    #     return {'label': single_prediction}


if __name__ == '__main__':
    valid_data=pd.read_csv(DATA_PATH+"/MedicalClass/v.csv")
    # valid_data.to_csv(DATA_PATH+"/MedicalClass/v.csv", index=0)
    data=valid_data['combined'].tolist()
    labels=valid_data['label'].tolist()
    prediction=Prediction()
    prediction.load_model()
    pred_labels=prediction.predict(data)
    count=0
    for label,pred_label in zip(labels,pred_labels):
        print(label)
        print(pred_label)
        if(label==pred_label[0][0]):
            count+=1
    print(len(labels))
    print(count)
    print('accuracy: {}'.format(count/len(labels)))
            
