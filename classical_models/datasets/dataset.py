#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 17:41:02 2023

@author: venkatesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 14:20:12 2021

@author: venkatesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 11:45:48 2021

@author: venkatesh
"""

import numpy as np
import pandas as pd
import scipy.io
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt





#%%
class mydataloader(Dataset):
    
    def __init__(self,  data_path,csv_for_patients,csv_for_images,training = True):
        self.data_path = data_path  
        self.training = training
        self.csv_for_patients=csv_for_patients
        self.csv_for_images=csv_for_images
        
        print('data path',self.data_path)
        
        # reading the patient details:
        self.patients_list = pd.read_csv(self.csv_for_patients)
        
        # reading the patient index and respective images indices....
        self.data=pd.read_csv(self.csv_for_images)

        print('patients_list_length:',len(self.patients_list))
        print('data_length:',len(self.data))
 
        print(self.patients_list)
        print(os.getcwd())

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        patient_id=self.data['patient_id'][idx]

        patient_file_name=self.patients_list['path'][patient_id]
        image_no=self.data['image_no'][idx]

        if self.training==True:
            try:
                
    
                #print('patient_id',patient_file_name,image_no)
    
    
                
                image_file_path=self.data_path+patient_file_name+"images/"+str(image_no)+".jpg"
                mask_file_path=self.data_path+patient_file_name+"masks/"+str(image_no)+".tif"
                
                #print('patient_image_path:',image_file_path)
                #print('patient_mask_path',mask_file_path)
                image_file=Image.open(image_file_path)
                mask_file=Image.open(mask_file_path)
    
                image_file=np.array(image_file)
                mask_file=np.array(mask_file)
                
                updated_image=image_file[4:452,11:331,:]
                updated_mask=mask_file[4:452,11:331]
                
                # making the numpy array into torch tensors.............
                updated_image=torch.from_numpy(updated_image)
                updated_mask=torch.from_numpy(updated_mask)
                
                updated_image=updated_image.permute(2,0,1)
                updated_mask=updated_mask.unsqueeze(0)
    
                # print('image_file.shape:',image_file.shape)
                # print('mask_file.shape:',mask_file.shape)
                
                return updated_image,updated_mask,patient_id,image_no
            except:
                print('exception:',patient_id,image_no)
                updated_image=np.zeros(shape=(464,352,3))
                updated_mask=np.zeros(shape=(464,352))
                
                updated_image=torch.from_numpy(updated_image)
                updated_mask=torch.from_numpy(updated_mask)
                
                updated_image=updated_image.permute(2,0,1)
                updated_mask=updated_mask.unsqueeze(0)
                

                return updated_image,updated_mask,patient_id,image_no
    
            # maing the image into numpy array

            

#%%
if __name__ == "__main__":

    # we are checking the data loader
    
    data_path='../data_making/aster_updated_data_nov_09_2022_with_flip/'
    data_path='../data_making/aster_updated_data_nov_09_2022_with_flip/'

    patients_path_csv=data_path+"/csv_files/patients_list_100.csv"

    batch_size=1


    tloader = mydataloader(data_path, '../data_making/aster_updated_data_nov_09_2022_with_flip/csv_files/full_data_csv/patients_list_1_99.csv', 
                           '../data_making/aster_updated_data_nov_09_2022_with_flip/csv_files/full_data_csv/data_with_v_h_1_79_train.csv')
    train_loader = DataLoader(tloader, batch_size = batch_size, shuffle=True, num_workers=1)
    
    
    vloader = mydataloader(data_path, '../data_making/aster_updated_data_nov_09_2022_with_flip/csv_files/full_data_csv/patients_list_1_99.csv', 
                           '../data_making/aster_updated_data_nov_09_2022_with_flip/csv_files/full_data_csv/data_with_v_h_80_89_validation.csv')
    val_loader = DataLoader(vloader, batch_size = batch_size, shuffle=True, num_workers=1)



    for i, data in enumerate(train_loader): 
            image_file,mask_file,patient_id,image_no=data
            print('image',image_file.shape,'mask',mask_file.shape)
    print('successfully read the trining loader.......')

