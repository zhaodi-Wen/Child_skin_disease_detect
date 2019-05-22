'''数据增强'''

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
from PIL import Image
# 数据生成器
dataGenerator = ImageDataGenerator(
    rotation_range=40,
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    channel_shift_range=0,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# 获取img下所有目录名
skin_dieases_list = os.listdir('./数据集/中文啥偶及的数据/new_img/儿童常见皮肤病图像集/')

def dataAugmentation(skin_dieases_list,dataGenerator):
    # 遍历皮肤病文件夹

    for skin_dieases in skin_dieases_list:
        path = './数据集/中文收集的数据/儿童常见皮肤病图像集/' + str(skin_dieases)  #图片路径,按类别放在对应的文件夹中
        # 获取每个皮肤病文件夹下的图片名字列表
        imglist = os.listdir(path)
        # 增加health皮肤
        for imgdir in imglist:
            img = load_img(path + '/' + imgdir)
            x_img = img_to_array(img)
            x_img = x_img.reshape((1,)+ x_img.shape)

            genData = dataGenerator.flow(
                x_img,
                batch_size=1,
                save_to_dir=path,
                save_format='jpg',
                save_prefix=str(skin_dieases)
            )
            for i in range(30):
                genData.next()
        print(skin_dieases+'增强完毕')

# 将所有图片resize成width*height
def getNormalImg(skin_dieases_list,width = 224,height = 224):

    for skin_dieases in skin_dieases_list:
        path = './数据集/中文收集的数据/new_img/儿童常见皮肤病图像集/' + str(skin_dieases)+'/'  #新的路径
        imglist = os.listdir(path)
        for imgdir in imglist:
            print(imgdir)
            img = Image.open(path+imgdir)
            new_img = img.resize((width,height),Image.ANTIALIAS)
            new_img.save(path+imgdir)
            new_img.close()
            print('成功规格化'+imgdir)


dataAugmentation(skin_dieases_list,dataGenerator)
getNormalImg(skin_dieases_list)
