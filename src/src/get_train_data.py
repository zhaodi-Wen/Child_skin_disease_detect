import os
import numpy as np
from keras.preprocessing.image import load_img,img_to_array,array_to_img
import re
import random
import
reg = '[^.]+'
imgPath = './img'
trainPath = './train/txt'

def get_train_data():

    skin_dieases_list = os.listdir(imgPath)
    for skin_dieases in skin_dieases_list:
        img_list = os.listdir(imgPath+'/'+str(skin_dieases))
        dir = trainPath+'/'+str(skin_dieases)
        if not os.path.exists(dir):
            os.mkdir(dir)
        for img in img_list:
            img_array = img_to_array(load_img(imgPath+'/'+str(skin_dieases)+'/'+str(img)))
            # 归一化
            img_array = img_array/255.
            img_array = np.resize(img_array,new_shape=(65536,3))
            fileName = re.match(reg, img).group() + '.txt'
            file = open(trainPath+'/'+str(skin_dieases)+'/'+str(fileName), 'w')
            np.savetxt(trainPath+'/'+str(skin_dieases)+'/'+str(fileName), img_array, fmt='%.8f', delimiter=' ')
            file.close()
        print(skin_dieases+'生成完毕')


get_train_data()


skin_dieases_list = os.listdir(imgPath)
# 随机选择一个皮肤病文件夹
skin_dieases_seed = random.randint(0, len(skin_dieases_list) - 1)
path = trainPath + '/' + skin_dieases_list[skin_dieases_seed]
train_list = os.listdir(trainPath + '/' + skin_dieases_list[skin_dieases_seed])
# 随机选择皮肤病文件夹下的一张图片
train_seed = random.randint(0, len(train_list) - 1)
x_input = np.loadtxt(trainPath + '/' + str(skin_dieases_list[skin_dieases_seed]) + '/' +
                  str(train_list[train_seed]))

x_input = np.resize(x_input,new_shape=(256,256,3))
print(x_input)

# 检查是否有nan
#np.any(np.isnan(x_input))

# label_file = re.match(reg, str(img_list[img_seed])).group() + '.txt'
# y_batch[i] = open(labelPath + '/' + str(skin_dieases_list[skin_dieases_seed]) + '/' +
#                  str(label_file)).readlines()