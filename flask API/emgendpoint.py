from flask import Flask,jsonify 
from PIL import Image
import base64
from io import BytesIO
import requests
import numpy as np
import pandas as pd
import pickle
import sklearn 
import json 
from sklearn.tree import ExtraTreeClassifier
model= ExtraTreeClassifier()
app = Flask(__name__)

#By using the <path: url> specifier, we ensure that the string that will come after send-image / is taken as a whole.
@app.route("/classify_emg/<path:url>")
def image_check(url):
   

    '''
    FUTURE PROCESS
    '''
    
    # When you type http://127.1.0.0:5000/send-image/https://sample-website.com/sample-cdn/photo1.jpg to the browser
    # you will se the whole "https://sample-website.com/sample-cdn/photo1.jpg"
    # return url
    # return jsonify({'amg':'sc'})
   

    class sEMGData:
        
        def __init__(self, path):
            self.path = path
            
            # file name
            f = open(path, 'r')
    
                
            # invalid_raise - sk
            # numpy array
        
            self.array = np.genfromtxt(self.path, skip_header=7, invalid_raise=False)
            if self.array.shape[1] == 6:
                 self.array = self.array[:,:5]
            # DataFrame
            self.dataframe = pd.DataFrame(self.array, columns=['RF','BF','VM','ST','FX'])
            
        def return_array(self):
            
            return self.array
        
        def return_df(self):
            
            return self.dataframe



    def sliding_window(df):
        
        # define the window size
        window_size = 5000

        # create empty container for rows of structured data
        structured_data = []

        # iterate over the time series data with a sliding window
        ## get percent of overlapping
        # overlapping = int(window_size * 0.9) = 4500
        for i in range(0,len(df.index) - window_size,500):
            window_start = i
            window_end = i + window_size
            # print('window_start',window_start,'end',window_end)

            # extract the window of data
            window = df[window_start:window_end]
            
            # print(len(window))
            # create the structured data row with five features
            row = {
                # RF      BF      VM      ST    FX
                'feature_one': window['RF'].pow(2).mean(),
                # 'feature_two': peak_vale(window,'RF'),
                'feature_two': window['RF'].max(),
                'feature_three': window['RF'].mean(),
                'feature_four': window['RF'].std(),
                'feature_five': window['RF'].var(),
                
                'feature2_one': window['BF'].pow(2).mean(),
                # 'feature2_two': peak_vale(window,'BF'),
                'feature2_two': window['BF'].max(),
                'feature2_three': window['BF'].mean(),
                'feature2_four': window['BF'].std(),
                'feature2_five': window['BF'].var(),
                
                'feature3_one': window['VM'].pow(2).mean(),
                # 'feature3_two': peak_vale(window,'VM'),
                'feature3_two': window['VM'].max(),
                'feature3_three': window['VM'].mean(),
                'feature3_four': window['VM'].std(),
                'feature3_five': window['VM'].var(),
                
                'feature4_one': window['ST'].pow(2).mean(),
                # 'feature4_two': peak_vale(window,'ST'),
                'feature4_two': window['ST'].max(),
                'feature4_three': window['ST'].mean(),
                'feature4_four': window['ST'].std(),
                'feature4_five': window['ST'].var(),

                'feature5_one': window['FX'].pow(2).mean(),
                # 'feature5_two': peak_vale(window,'FX'),
                'feature5_two':  window['FX'].max(),
                'feature5_three': window['FX'].mean(),
                'feature5_four': window['FX'].std(),
                'feature5_five': window['FX'].var()
            }
            
            # add the row to the container
            structured_data.append(row)

            # convert the container into a dataframe
            df_structured = pd.DataFrame(structured_data)

        return df_structured
    file_path = f'/workspaces/Save_knee/abnormal test//{url}'
    
    sd = sEMGData(file_path)
    df = sd.return_df()
    structured = sliding_window(df)
    

    # print(structured.head(5))
    model = pickle.load(open('/workspaces/Save_knee/emg/EMG4500.pkl','rb'))
    # df1 = pd.DataFrame(structured)

    structured.insert(0,'index',0)
    X = structured.iloc[0]
    pre_prob = model.predict_proba([X])
    y_pred = model.predict([X])
    y_pred = y_pred.tolist()
    class_name = ''
    if y_pred[0] == 0:
        class_name = 'Normal'
    else:
        class_name = 'Abnormal'

    responseA = json.dumps({'prediction':y_pred,
                      'prediction_proba': pre_prob.tolist(),
                      'prediction_name' : class_name
                      })
    return responseA
    


if __name__ == '__main__':
    app.run(debug=True)