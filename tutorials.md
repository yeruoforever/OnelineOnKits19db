# 一行代码处理kits19数据集






```python
from kits19tools import *
data_dir="D:/kits19/data"
```

## 读取3D图像和标注


```python
case_id=10
img=get_imaging(data_dir,case_id)
seg=get_segmentation(data_dir,case_id)
[img,seg]
```




    [<nibabel.nifti1.Nifti1Image at 0x20d01163f88>,
     <nibabel.nifti1.Nifti1Image at 0x20d01175408>]



## 两肾可视化


```python
visualization(data_dir,case_id,resample_1_1=True,clip=array([-30,280]))
```


![svg](tutorials_files/tutorials_5_0.svg)


## 获取图像信息
* 3D图像数组大小
* 体素间距（mm）


```python
size=get_size(img)
space=get_spacing(img)
print(size,space)
```

    [ 50 512 512] [3.        0.7578125 0.7578125]
    

## 分析前景位置信息

* 前景3D切片范围（一个立方体，同时包含两个肾脏和肿瘤）
* 肾脏和肿瘤3D切片范围（两个立方体，包含两个肾脏的立方体和包含所有肿瘤的立方体）
* 左右肾脏3D切片范围（两个立方体，分别为左右两肾）
* 左右两肾脏的质心坐标


```python
print("front",analyze_cube(seg))
for cube,name in zip(analyze_cubes(seg),["kidney","tumour"]):
    print(name,cube)

for i,center in enumerate(analyze_kidney_center(seg)):
    print("kidney %d center"%i,center)

kidneys,centers=analyze_kidney(seg)
for i,(kidney,center) in enumerate(zip(kidneys,centers)):
    print("kidney %o center"%i,center,"range",kidney)

```

    front (slice(9, 50, None), slice(226, 352, None), slice(132, 380, None))
    kidney (slice(9, 50, None), slice(226, 352, None), slice(132, 380, None))
    tumour (slice(10, 20, None), slice(265, 311, None), slice(290, 331, None))
    kidney 0 center [ 22 296 337]
    kidney 1 center [ 31 280 170]
    kidney 0 center [ 31 280 170] range (slice(15, 49, None), slice(226, 339, None), slice(132, 214, None))
    kidney 1 center [ 22 296 337] range (slice(9, 39, None), slice(246, 351, None), slice(290, 379, None))
    

## 重采样


```python
space=array([1.62,1.62,1.62])

img_size=get_size(img)
seg_size=get_size(seg)

img_space=get_spacing(img)
seg_space=get_spacing(seg)

print("img",img_size,img_space)
print("seg",seg_size,seg_space)

img_resample=resample_image(img,space)
seg_resample=resample_segmentation(seg,space)

img_size=get_size(img_resample)
seg_size=get_size(seg_resample)

img_space=get_spacing(img_resample)
seg_space=get_spacing(seg_resample)

print("img",img_size,img_space)
print("seg",seg_size,seg_space)

img_data=img.get_fdata()
print(img_data.shape)
space_new=array([3,3,3])
img_data_resample=resample_arrray(img_data,img_space,space_new)
print(img_data_resample.shape)
```

    img [ 50 512 512] [3.        0.7578125 0.7578125]
    seg [ 50 512 512] [3.        0.7578125 0.7578125]
    img [ 93 240 240] [1.62 1.62 1.62]
    seg [ 93 240 240] [1.62 1.62 1.62]
    (50, 512, 512)
    (27, 276, 276)
    

## 保存重采样图像和标注


```python
path_sample="D:/tmp/"

save_imaging(path_sample,case_id,img_resample)
save_segmentation(path_sample,case_id,seg_resample)
```

## 统计像素信息
* 统计单个病例
* 统计整个数据集


```python
mean,std=analyze_mean_std(img)
print(mean,std)
```

    -479.09187461853026 479.0315006505917
    


```python
cases=list(range(30))
mean,std,nums=statistics(data_dir,cases)
print("mean:",mean)
print("std",std)
print("nums",nums)
```

    mean: -527.8739892580843
    std 2900307.079352456
    nums 1766326272
    


```python

```
