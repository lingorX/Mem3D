
import numpy as np
import os                #遍历文件夹
import nibabel as nib    #nii格式一般都会用到这个包
import cv2           #转换成图像
from PIL import Image 
def nii_to_image(niifile):
    filenames = os.listdir(filepath)  #读取nii文件夹
    slice_trans = []
 
    for f in filenames:
        #开始读取nii文件
        img_path = os.path.join(filepath, f ,'segmentation.nii')
        img = nib.load(img_path)                #读取nii
        img_fdata = img.get_fdata()
        fname = f           #去掉nii的后缀名
        img_f_path = os.path.join(imgfile, fname)
        #创建nii对应的图像的文件夹
        if not os.path.exists(img_f_path):
            os.mkdir(img_f_path)                #新建文件夹
 
        #开始转换为图像
        (x,y,z) = img.shape
        for i in range(x): #z是图像的序列
            silce = img_fdata[i, :, :]        #选择哪个方向的切片都可以
            im = Image.fromarray(silce).convert('P')
            im.putpalette([0,0,0,255,0,0,0,255,0])
            im.save(os.path.join(img_f_path,'{}.png'.format(i)), format='PNG')
 
if __name__ == '__main__':
    filepath = '/home/lll/liliulei/data'
    imgfile = '/home/lll/liliulei/mask'
    nii_to_image(filepath)
