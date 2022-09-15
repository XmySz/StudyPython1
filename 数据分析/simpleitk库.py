"""
    Insight Segmentation and Registration Toolkit (ITK)是一个开源、跨平台的框架，可以提供给开发者增强功能的图像分析和处理套件。
其中最为著名的就是SimpleITK，是一个简化版的、构建于ITK最顶层的模块。SimpleITK旨在易化图像处理流程和方法。

"""
import SimpleITK as sitk

dir_path = "/media/sci/KE002/6-ROI/sub001/T2"
path = "/media/sci/KE002/6-ROI/sub001/T2tumor.mha"

# 读取操作
itk_img = sitk.ReadImage(path)  # 读取3d图像,格式一般为mhd,dicom,mha,nii.gz
img_array = sitk.GetArrayFromImage(itk_img)  # 转换为np对象
itk_img_copy = sitk.GetImageFromArray(img_array)  # 将np对象转换为3d图像对象，返回一个副本
seriresReader = sitk.ImageSeriesReader()  # 读取dicom序列数据
seriesIds = seriresReader.GetGDCMSeriesIDs(dir_path)
filenames = seriresReader.GetGDCMSeriesFileNames(dir_path, seriesIds[0])
seriresReader.SetFileNames(filenames)
dicom_image = seriresReader.Execute()

# 保存操作
# sitk.WriteImage(itk_img, fileName="")

# 相关属性
print(img_array.shape)  # 3d图像的形状
print(itk_img.GetSize())  # 3d图像的形状
print(itk_img.GetOrigin())  # 3d图像的坐标原点
print(itk_img.GetDirection())  # 3d图像的方向（物理空间上的轴线方向）
print(itk_img.GetSpacing())  # 3d图像的尺度信息(像素之间的实际距离)
