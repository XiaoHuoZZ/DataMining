import os
import xml.dom.minidom as xl
import re
import csv
import FileUtils



def sougou_handle(path):
    dom = xl.parse(path)
    root = dom.documentElement
    docs = root.getElementsByTagName('doc')
    category = []
    i = 0
    for doc in docs:
        i = i + 1
        url = doc.getElementsByTagName('url')[0].firstChild.data
        cty =  re.match(r'http://([0-9a-zA-Z]+)',url).group(1) 
        title_obj = doc.getElementsByTagName('contenttitle')[0].firstChild
        if title_obj is not None:
            title = title_obj.data
        else:
            title = ''
        content_obj = doc.getElementsByTagName('content')[0].firstChild
        if content_obj is not None:
            content = content_obj.data
        else:
            continue
        if cty not in category:
            category.append(cty)
        with open('./data/data.csv', 'a', newline='',encoding='utf-8') as f: 
            row = []
            row.append(cty)
            row.append(title)
            row.append(content)
            writer = csv.writer(f) 
            writer.writerow(row) 
            f.close 
    return category,i

def sougou_pre(path):
    with open(path,'r',encoding='utf-8') as f:
        s = f.read()
        ns = s.replace('&','&amp;')
        res = '<docs>\n' + ns + '</docs>'
        f.close
    with open(path,'w',encoding='utf-8') as f:
        f.write(res)
        f.close
    return

rootdir = '.\sougou'
dirs = os.listdir(rootdir) #列出文件夹下所有的目录与文件



# 转换为utf-8格式
# for i in range(0,len(dirs)):
#     FileUtils.convert(os.path.join(rootdir,dirs[i]),in_enc='gbk')

 #首次创建添加表头
with open('./data/data.csv', 'w', newline='',encoding='utf-8') as f: 
    head = ['category', 'title','content']
    writer = csv.writer(f) 
    writer.writerow(head) 
    f.close()



category = []
size = 0

for i in range(0,len(dirs)):
    path = os.path.join(rootdir,dirs[i])
    #更正xml 只能首次
    sougou_pre(path)
    c,s = sougou_handle(path)
    category = list(set(category + c))
    size = size + s
print(category)
print(size)
