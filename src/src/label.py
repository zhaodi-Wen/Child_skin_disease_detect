import numpy as np
import os
import re
from keras.preprocessing.image import img_to_array
# 获取img下所有目录名
skin_dieases_list = os.listdir('./new_img')
labelPath = './new_label'
if not os.path.exists(labelPath):
    os.mkdir(labelPath)

def getLabelData(skin_dieases_list):
    dims = len(skin_dieases_list)
    reg = '[^.]+'
    i = 0 # 计录是第几类皮肤病
    for skin_dieases in skin_dieases_list:

        imgPath = './new_img/' + str(skin_dieases)
        imgFileNameList = os.listdir(imgPath)
        dir = labelPath +'/'+ str(skin_dieases)
        os.makedirs(dir)
        for imgFileName in imgFileNameList:
            fileName = re.match(reg,imgFileName).group()+'.txt'
            array = np.zeros((dims,1),int)
            array[i,0] = 1
            file = open(labelPath+'/'+str(skin_dieases)+'/'+str(fileName),'w')
            np.savetxt(labelPath+'/'+str(skin_dieases)+'/'+str(fileName),array)
            file.close()

        print(skin_dieases+'标签生成完毕')
        i+=1


getLabelData(skin_dieases_list)