
import cv2
import numpy as np
import glob

img_array = []


image_list=glob.glob('savedModels/full_data_experiemnts/muti_bce_loss_fusion/Jul_29_06_42_pm_data_representation_model_u2net/results_5/both_mask_on_image/patient_1/*.png')
image_list.sort()
print(len(image_list))

for filename in image_list:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

    

out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()