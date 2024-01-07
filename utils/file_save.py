import os
import xlwt
import xlrd
import datetime
import time
import sys
import openpyxl

log_path = "../logs_convnext_drop/"


def get_filelist(dir, Filelist):
    newDir = dir
    if os.path.isfile(dir):
        Filelist.append(os.path.basename(dir))
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            # 如果需要忽略某些文件夹，使用以下代码
            # if s == "xxx":
            # continue
            newDir = os.path.join(dir, s)
            get_filelist(newDir, Filelist)
    return Filelist


list_y = []
list = get_filelist(log_path, list_y)
filenames = "data_Record(2)"


def txtconvert(filename, Filenames):
    global f
    xls = xlwt.Workbook()
    p = 0
    xls_data = xls.add_sheet('sheet1', cell_overwrite_ok=True)
    xls_data.write(0, 0, "prompts")
    xls_data.write(0, 1, "traits")
    xls_data.write(0, 2, "best epoch")
    xls_data.write(0, 3, "best val")
    xls_data.write(0, 4, "best test")
    x12 = 1

    xls_data.write(0, 8, "prompts")
    xls_data.write(0, 9, "traits")
    xls_data.write(0, 10, "best epoch")
    xls_data.write(0, 11, "best val")
    xls_data.write(0, 12, "best test")
    x36 = 1

    xls_data.write(0, 16, "prompts")
    xls_data.write(0, 17, "traits")
    xls_data.write(0, 18, "best epoch")
    xls_data.write(0, 19, "best val")
    xls_data.write(0, 20, "best test")
    x7 = 1

    xls_data.write(0, 24, "prompts")
    xls_data.write(0, 25, "traits")
    xls_data.write(0, 26, "best epoch")
    xls_data.write(0, 27, "best val")
    xls_data.write(0, 28, "best test")
    x8 = 1

    for i in filename:
        txtfile = filename
        filepathname = str(list[p]).replace("epoch_QWK", "loss")
        filepathname1 = filepathname.replace(".txt", "")
        filepanth = (log_path + list[p])
        filepathnameture = str((log_path + filepathname1 + "/" + list[p]))
        f = open(filepathnameture, "r")
        list11 = []
        y12 = 0
        y36 = 8
        y7 = 16
        y8 = 24
        z12 = 0
        z36 = 0
        z7 = 0
        z8 = 0
        while True:  # 循环，读取文本里面的所有内容
            line = f.readline()  # 一行一行读取
            if not line:  # 如果没有内容，则退出循环
                break
            list11.append(line)
            asdd = list11[0]
            the_features = int(asdd)
            if the_features == 1:
                print("the 1")
                for i in line.split('\t'):  # 读取出相应的内容写到x
                    item = i.strip()
                    xls_data.write(x12, y12, item)
                    y12 = y12 + 1  # 另起一列
                    z12 = z12 + 1

            if the_features == 2:
                print("the 2")
                for i in line.split('\t'):  # 读取出相应的内容写到x
                    item = i.strip()
                    xls_data.write(x12, y12, item)
                    y12 += 1  # 另起一列
                    z12 = z12 + 1

            if the_features == 3:
                for i in line.split('\t'):  # 读取出相应的内容写到x
                    item = i.strip()
                    xls_data.write(x36, y36, item)
                    y36 += 1  # 另起一列
                    z36 = z36 + 1
            if the_features == 4:
                for i in line.split('\t'):  # 读取出相应的内容写到x
                    item = i.strip()
                    xls_data.write(x36, y36, item)
                    y36 += 1  # 另起一列
                    z36 = z36 + 1
            if the_features == 5:
                for i in line.split('\t'):  # 读取出相应的内容写到x
                    item = i.strip()
                    xls_data.write(x36, y36, item)
                    y36 += 1  # 另起一列
                    z36 = z36 + 1
            if the_features == 6:
                for i in line.split('\t'):  # 读取出相应的内容写到x
                    item = i.strip()
                    xls_data.write(x36, y36, item)
                    y36 += 1  # 另起一列
                    z36 = z36 + 1
            if the_features == 7:
                for i in line.split('\t'):  # 读取出相应的内容写到x
                    item = i.strip()
                    xls_data.write(x7, y7, item)
                    y7 += 1  # 另起一列
                    z7 = z7 + 1
            if the_features == 8:
                for i in line.split('\t'):  # 读取出相应的内容写到x
                    item = i.strip()
                    xls_data.write(x8, y8, item)
                    y8 += 1  # 另起一列
                    z8 = z8 + 1
        if z12 == len(list11):
            x12 = x12 + 1
        if z36 == len(list11):
            x36 = x36 + 1
        if z7 == len(list11):
            x7 = x7 + 1
        if z8 == len(list11):
            x8 = x8 + 1

        p = p + 1
    f.close()
    xls.save(Filenames + '.xls')  # 保存


txtconvert(list, filenames)
if __name__ == "__file_save__":
    filename = sys.argv[1]
    xlsname = sys.argv[2]
    txtconvert(filename, xlsname)
