import urllib.request
import re
import os
'''·爬虫爬取图像,获取数据集'''
#--------------------
# skin_dieases_url_list
# skin_dieases_list
#--------------------


# 获取网页源代码
def getHtml(url):
    page = urllib.request.urlopen(url)
    html = page.read()
    return html

# 通过正则表达式获取列表
def getList(reg,html):
    ojre = re.compile(reg)
    ojList = re.findall(ojre,str(html))
    return ojList

# 主页
url = "https://medicine.uiowa.edu/dermatology/education/clinical-skin-disease-images"
# 匹配皮肤病列表正则
skin_dieases_reg = 'a href="//medicine.uiowa.edu/dermatology/(.[^"]+)">'
main_html = getHtml(url)
skin_dieases_list = getList(skin_dieases_reg,main_html)

# 获取各个皮肤病的图片网页
skin_dieases_url_list = []
for skin_dieases in skin_dieases_list:
    skin_dieases_url_list.append('https://medicine.uiowa.edu/dermatology/' + skin_dieases)

# 进入各个皮肤病网页并下载图片
def getImg(skin_dieases_url_list,skin_dieases_list):
    imgre = re.compile('typeof="foaf:Image" src="(.+?\.jpg)" alt')
    i=0 #记录当前是第几个皮肤病
    for skin_dieases_url in skin_dieases_url_list:
        # 当前皮肤病名字
        skin_dieases_current = str(skin_dieases_list[i])
        skin_dieases_html = getHtml(skin_dieases_url)
        imglist = re.findall(imgre,str(skin_dieases_html))
        # print(imglist)
        # 下载路径
        dir = '数据集/爬虫爬取的数据/new_img/儿童常见皮肤病图像集/' + skin_dieases_current
        os.makedirs(dir)
        print(skin_dieases_current+'文件夹创建成功')
        print(imglist)
        for index in range(0,len(imglist)):
            picname = skin_dieases_current+str(index+1)+'.jpg'
            filename = os.path.join(dir,picname)
            urllib.request.urlretrieve('https:'+str(imglist[index]), filename)
            print(filename+'下载完成')

        i+=1

getImg(skin_dieases_url_list,skin_dieases_list)









