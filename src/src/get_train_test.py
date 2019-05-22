import os
import shutil
import re

img_path = './new_img'
label_path = './new_label'
test_path = './test_new'
train_path = './train_new'
train_label = train_path+'/label'
train_img = train_path+'/img'
test_label = test_path + '/label'
test_img = test_path+'/img'
train_percent = 0.7
test_percent = 0.3
reg = '[^.]+'
path = [train_path,train_img,train_label,
        test_path,test_img,test_label]
for sub_path in path:
    if not os.path.exists(sub_path):
        os.mkdir(sub_path)
skin_dieases_list = os.listdir('./new_img')
test_num = len(skin_dieases_list)*test_percent
train_num = len(skin_dieases_list)*train_percent


for i in range(len(skin_dieases_list)):
    img_list = os.listdir(img_path+'/'+str(skin_dieases_list[i]))
    print(img_list)
    for j in range(int(len(img_list)*train_percent)):
        img_source = img_path+'/'+str(skin_dieases_list[i])+'/'+str(img_list[j])
        img_dst = train_img+'/'+str(skin_dieases_list[i])
        label_name = re.match(reg,str(img_list[j])).group()+'.txt'
        label_source = label_path+'/'+str(skin_dieases_list[i])+'/'+str(label_name)
        #print('label ',label_source)
        label_dst = train_label+'/'+str(skin_dieases_list[i])
        if not os.path.exists(img_dst) :
            os.mkdir(img_dst)
        if not os.path.exists(label_dst):
            os.mkdir(label_dst)
        shutil.copy(img_source,img_dst)
        #shutil.copy(label_source,label_dst)


    for k in range(int(len(img_list)*train_percent+1),len(img_list)):
        img_source = img_path+'/'+str(skin_dieases_list[i])+'/'+str(img_list[k])
        img_dst = test_img+'/'+str(skin_dieases_list[i])

        label_name = re.match(reg,str(img_list[k])).group()+'.txt'
        label_source = label_path+'/'+str(skin_dieases_list[i])+'/'+str(label_name)
        label_dst = test_label+'/'+str(skin_dieases_list[i])
        if not os.path.exists(img_dst):
            os.mkdir(img_dst)
        if not os.path.exists(label_dst):
            os.mkdir(label_dst)

        shutil.copy(img_source,img_dst)
        #shutil.copy(label_source,label_dst)





