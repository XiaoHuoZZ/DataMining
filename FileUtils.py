import sys
import codecs
import chardet


def convert(file, in_enc="gb18030", out_enc="UTF-8"):
    """
    该程序用于将目录下的文件从指定格式转换到指定格式，默认的是GBK转到utf-8
    :param file:    文件路径
    :param in_enc:  输入文件格式
    :param out_enc: 输出文件格式
    :return:
    """
    in_enc = in_enc.upper()
    out_enc = out_enc.upper()
    try:
        print("convert [ " + file.split('\\')[-1] + " ].....From " + in_enc + " --> " + out_enc )
        f = codecs.open(file, 'r', in_enc ,errors='ignore')
        new_content = f.read()
        codecs.open(file, 'w', out_enc).write(new_content)
    # print (f.read())
    except IOError as err:
        print("I/O error: {0}".format(err))
def detect(file):
    # 我要打开的是二进制的文件，所用的是rb
    f = open(file,'rb')
    data = f.read()
    f.close()
    return chardet.detect(data)['encoding']