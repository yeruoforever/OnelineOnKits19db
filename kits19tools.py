from os.path import join,exists
from os import makedirs
import json
from functools import reduce
from typing import Optional

from nibabel import load,save,Nifti1Image
from numpy import ndarray,array,clip,ceil,matmul,diag,around,power,argwhere,diff,split,where,empty,append
from scipy import ndimage as ndimage
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

KINDEY = 1
TUMOUR = 2

def try_case_dir(path:str,case_id:int):
    '''测试`case_id`的文件夹是否存在，不存在则创建它'''
    case_dir=join(path,"case_%05d"%case_id)
    if not exists(case_dir):
        makedirs(case_dir)

def get_imaging_path(data_dir:str,case_id:int)->str:
    '''获得nii.gz图像文件路径
    args:
        data_dir:str    数据集根目录
        case_id:int     病例id
    return:
        path    图像文件路径
    '''
    return join(data_dir,"case_%05d"%case_id,"imaging.nii.gz")

def get_segmentation_path(data_dir:str,case_id:int)->str:
    '''获得分割标注路径
    args:
        data_dir:str    数据集根目录
        case_id:int     病例id
    return:
        path    分割标注文件路径
    '''
    return join(data_dir,"case_%05d"%case_id,"segmentation.nii.gz")

def get_imaging(data_dir:str,case_id:int)->Nifti1Image:
    '''获得影像对象
    args:
        data_dir:str    数据集根目录
        case_id:int     病例id
    return:
        img:Nifti1Image CT图像数据对象
    '''
    path=get_imaging_path(data_dir,case_id)
    return load(path)

def get_segmentation(data_dir:str,case_id:int)->Nifti1Image:
    '''获得分割标注对象
    args:
        data_dir:str    数据集根目录
        case_id:int     病例id
    return:
        seg:Nifti1Image 分割标注对象
    '''
    path=get_segmentation_path(data_dir,case_id)
    return load(path)

def save_imaging(data_dir:str,case_id:int,image:Nifti1Image)->None:
    '''保存影像对象至文件
    args:
        data_dir:str    数据集根目录
        case_id:int     病例id
        image：Nifti1Image  待保存对象
    '''
    try_case_dir(data_dir,case_id)
    path=get_imaging_path(data_dir,case_id)
    save(image,path)

def save_segmentation(data_dir:str,case_id:int,segmentation:Nifti1Image)->None:
    '''保存分割标注对象至文件
    args:
        data_dir:str    数据集根目录
        case_id:int     病例id
        segmentation：Nifti1Image  待保存对象
    '''
    try_case_dir(data_dir,case_id)
    path=get_segmentation_path(data_dir,case_id)
    save(segmentation,path)

def get_spacing(img:Nifti1Image)->ndarray:
    '''获取影像体素点间的体素间距
    args：
        img:Nifti1Image
    return:
        spacing:ndarray [d,w,h]
    '''
    return array(img.header.get_zooms())

def get_size(img:Nifti1Image)->ndarray:
    '''获取影像数据维度大小
    args：
        img:Nifti1Image
    return:
        size:ndarray [d,w,h]
    '''
    return array(img.shape)

def get_data(img:Nifti1Image)->ndarray:
    '''获取影像数据
    args：
        img:Nifti1Image
    return:
        data:ndarray size[d,w,h]
    '''
    return img.get_fdata()

def resample_size(size:ndarray,space_origin:ndarray,space_target:ndarray=array([3.22,1.62,1.62]))->ndarray:
    # 粗略估计
    '''估计重采样后的影像数据大小（各维度误差不超过±1）
    args:
        size:ndarray    原始大小
        space_origin    原始间距
        space_target    目标间距
    return：
        size:ndarray    重采样后的影像数据大小
    '''
    return ceil(space_origin/space_target*size).astype(int)

def resample_arrray(np_img:ndarray,space_origin:ndarray,space_target:ndarray=array([3.22,1.62,1.62]))->ndimage:
    '''对影像数据进行重采样（类型为numpy.ndarray）
    args:
        np_img:ndarray  原始数组
        space_origin    原始间距
        space_target    目标间距
    return：
        data:ndarray    重采样后的影像数据
    '''
    scales=space_origin/space_target
    return ndimage.interpolation.zoom(np_img,scales)

def resample_image(img:Nifti1Image,space_target:ndarray=array([3.22,1.62,1.62]))->Nifti1Image:
    '''对影像数据进行重采样（类型为Nifti1Image）
    args:
        img:Nifti1Image 原始数组
        space_target    目标间距
    return：
        data:Nifti1Image    重采样后的影像数据
    '''
    aff=-diag([*space_target,-1])
    aff=matmul(array([
        [0,0,1,0],
        [0,1,0,0],
        [1,0,0,0],
        [0,0,0,1],
    ]),aff)
    scales=get_spacing(img)/space_target
    data=img.get_fdata()
    resample=ndimage.zoom(data,scales,mode="reflect")
    return Nifti1Image(resample,aff)

def resample_segmentation(seg:Nifti1Image,space_target:ndarray=array([3.22,1.62,1.62]))->Nifti1Image:
    '''对分割标注数据进行重采样（类型为Nifti1Image）
    args:
        img:Nifti1Image 原始影像
        space_target    目标间距
    return：
        data:Nifti1Image    重采样后的分割标注数据
    '''
    aff=-diag([*space_target,-1])
    aff=matmul(array([
        [0,0,1,0],
        [0,1,0,0],
        [1,0,0,0],
        [0,0,0,1],
    ]),aff)
    scales=get_spacing(seg)/space_target
    data=seg.get_fdata()
    resample=ndimage.interpolation.zoom(data,scales,order=1,mode="reflect")
    resample=around(resample,0).astype(int)
    return Nifti1Image(resample,aff)

def analyze_mean_std(img:Nifti1Image):
    '''分析影像体素点的均值和方差
    args:
        img:Nifti1Image 待分析影像
    return：
        mean:float  均值
        std:float   方差
    '''
    data=img.get_fdata()
    mean=data.mean()
    std=data.std()
    return mean,std

def analyze_cube(seg:Nifti1Image)->tuple:
    '''粗分割（前景）体素点在数组中的分布范围
    args:
        seg:Nifti1Image 体素点标注
    return：
        slice_tuple:tuple(slice,N)  各个维度上的范围切片对象，N为维度 
    '''
    indexs=argwhere(seg.get_fdata()!=0)
    return tuple(slice(start,end+1) for start,end in zip(indexs.min(axis=0),indexs.max(axis=0)))

def analyze_cubes(seg:Nifti1Image)->list:
    '''细分割（每种前景）体素点在数组中的分布范围
    args:
        seg:Nifti1Image 体素点标注
    return：
        list:list[tuple(slice,N)...]    每种前景对象在各个维度上的范围切片对象，N为维度 
    '''
    return ndimage.find_objects(seg.get_fdata().astype(int))
    
def analyze_kidney_center(seg:Nifti1Image)->list:
    '''每个肾脏在3D图像中坐标的质心
    args:
        seg:Nifti1Image 体素点标注
    return:
        list:[[d,w,h],[d,w,h]]
    '''
    indexs=argwhere(seg.get_fdata()>0)
    kmeans=KMeans(n_clusters=2,precompute_distances=True).fit(indexs)
    return array(kmeans.cluster_centers_,dtype=int)

def analyze_kidney(seg:Nifti1Image,align=False)->list:
    '''每个肾脏在数组中的体素点分布范围
    args:
        seg:Nifti1Image 体素点标注
        align:bool      两肾选框是否对其
    return：
        list:list[tuple(slice,N)...]    两肾在各个维度上的范围切片对象，N为维度 
        centers:ndarray 两肾质心坐标
    '''
    data=seg.get_fdata()
    indexs=argwhere(data>0)
    kmeans=KMeans(n_clusters=2,precompute_distances=True).fit(indexs)
    
    if align:
        ranges=[]
        for center in kmeans.cluster_centers_:
            d,w,h=center
            d_min,d_max=d-1,d+1
            w_min,w_max=w-1,w+1
            h_min,h_max=h-1,h+1
            while d_min >0:
                if KINDEY not in data[d_min,:,:]:
                    break
                d_min-=1
            while w_min >0:
                if KINDEY not in data[:,w_min,:]:
                    break
                w_min-=1
            while h_min >0:
                if KINDEY not in data[:,:,h_min]:
                    break
                h_min-=1
            while d_max < data.shape[0]:
                if KINDEY not in data[d_max,:,:]:
                    break
                d_max+=1
            while w_max < data.shape[1]:
                if KINDEY not in data[:,w_max,:]:
                    break
                w_max+=1
            while h_max <data.shape[2]:
                if KINDEY not in data[:,:,h_max]:
                    break
                h_max+=1
            ranges.append((slice(d_min,d_max),slice(w_min,w_max),slice(h_min,h_max)))
        return ranges,array(kmeans.cluster_centers_,dtype=int)
    else:

        labels=kmeans.predict(indexs)
        kindey1=indexs[labels==0]
        kindey2=indexs[labels==1]

        d_min_1,w_min_1,h_min_1=kindey1.min(axis=0)
        d_max_1,w_max_1,h_max_1=kindey1.max(axis=0)
        d_min_2,w_min_2,h_min_2=kindey2.min(axis=0)
        d_max_2,w_max_2,h_max_2=kindey2.max(axis=0)
        return [
            (slice(d_min_1,d_max_1),slice(w_min_1,w_max_1),slice(h_min_1,h_max_1)),
            (slice(d_min_2,d_max_2),slice(w_min_2,w_max_2),slice(h_min_2,h_max_2))
        ],array(kmeans.cluster_centers_,dtype=int)

def statistics(data_dir:str,cases:Optional[list]=None)->tuple:
    '''统计所有病例影像中体素点的均值、方差和总数
    args:
        data_dir:str 数据集根目录
    return：
        mean:float  均值
        std:float   方差
        count：int  体素点个数
    '''
    # 原始数据输出为 -527.4219503457223 299251.51932233473 17118994432
    mean_all=[]
    size_all=[]

    if cases is None:
        cases=list(range(300))
    
    for case_id in cases:
        img=get_imaging(data_dir,case_id)
        mean,std=analyze_mean_std(img)
        size=get_size(img)
        size=reduce(lambda x,y:x*y,size)
        mean_all.append(mean)
        size_all.append(size)

    count=sum(size_all)
    mean=sum(a*n for a,n in zip(mean_all,size_all))/count

    sum_std=0.

    for case_id in range(300):
        img=get_imaging(data_dir,case_id)
        data=img.get_fdata()
        part_std=power(data-mean,2)/count
        sum_std+=part_std.sum()

    std=sum_std

    return mean,std,count

def statistics_large_memory(data_dir)->tuple:
    '''统计所有病例影像中体素点的均值、方差和总数（已弃用），完整运行大约需要338GB内存空间
    args:
        data_dir:str 数据集根目录
    return：
        mean:float  均值
        std:float   方差
        count：int  体素点个数
    '''
    # 完整运行大约需要338GB内存空间
    values=empty(shape=(0,512,512),dtype=float)
    for case_id in range(300):
        img=get_imaging(path,case_id)
        values=append(values,img.get_fdata())
    return values.mean(),values.std(),reduce(lambda x,y:x*y,values.shape)

def visualization(path:str,case_id:int,resample_1_1:bool=False,clip=[-30,300]):
    '''以每个肾脏的质心所在位置，绘制三视图
    args:
        case_id:int         病例id
        resample_1_1:bool   是否按照d:w:h=1:1:1可视化
        clip:ndarray        像素值截断区间[val_min,val_max]   

    '''
    img=get_imaging(path,case_id)
    seg=get_segmentation(path,case_id)
    if resample_1_1:
        img=resample_image(img,array([1.62,1.62,1.62]))
        seg=resample_segmentation(seg,array([1.62,1.62,1.62]))

    kindeys,centers=analyze_kidney(seg)
    
    img_data=img.get_fdata().clip(*clip)

    fig=plt.subplots(figsize=(16,8))
    ax=plt.subplot2grid((2,4),(0,0))
    ax.imshow(img_data[:,centers[0][1],:],cmap=plt.cm.gray)
    ax.add_patch(
        patches.Rectangle(
            (kindeys[0][2].start,kindeys[0][0].start),
            kindeys[0][2].stop-kindeys[0][2].start,
            kindeys[0][0].stop-kindeys[0][0].start,
            linewidth=2,edgecolor='r',facecolor='none'
        )
    )
    ax=plt.subplot2grid((2,4),(0,1))
    ax.imshow(img_data[:,:,centers[0][2]],cmap=plt.cm.gray)
    ax.add_patch(
        patches.Rectangle(
            (kindeys[0][1].start,kindeys[0][0].start),
            kindeys[0][1].stop-kindeys[0][1].start,
            kindeys[0][0].stop-kindeys[0][0].start,
            linewidth=2,edgecolor='r',facecolor='none'
        )
    )
    ax=plt.subplot2grid((2,4),(1,0))
    ax.imshow(img_data[centers[0][0],:,:],cmap=plt.cm.gray)
    ax.add_patch(
        patches.Rectangle(
            (kindeys[0][2].start,kindeys[0][1].start),
            kindeys[0][2].stop-kindeys[0][2].start,
            kindeys[0][1].stop-kindeys[0][1].start,
            linewidth=2,edgecolor='r',facecolor='none'
        )
    )
    ax=plt.subplot2grid((2,4),(1,1),projection="3d")
    p=img_data[kindeys[0]]
    verts, faces ,_,_= measure.marching_cubes(p)
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    ax=plt.subplot2grid((2,4),(0,2))
    ax.imshow(img_data[:,centers[1][1],:],cmap=plt.cm.gray)
    ax.add_patch(
        patches.Rectangle(
            (kindeys[1][2].start,kindeys[1][0].start),
            kindeys[1][2].stop-kindeys[1][2].start,
            kindeys[1][0].stop-kindeys[1][0].start,
            linewidth=2,edgecolor='r',facecolor='none'
        )
    )
    ax=plt.subplot2grid((2,4),(0,3))
    ax.imshow(img_data[:,:,centers[1][2]],cmap=plt.cm.gray)
    ax.add_patch(
        patches.Rectangle(
            (kindeys[1][1].start,kindeys[1][0].start),
            kindeys[1][1].stop-kindeys[1][1].start,
            kindeys[1][0].stop-kindeys[1][0].start,
            linewidth=2,edgecolor='r',facecolor='none'
        )
    )
    ax=plt.subplot2grid((2,4),(1,2))
    ax.imshow(img_data[centers[1][0],:,:],cmap=plt.cm.gray)
    ax.add_patch(
        patches.Rectangle(
            (kindeys[1][2].start,kindeys[1][1].start),
            kindeys[1][2].stop-kindeys[1][2].start,
            kindeys[1][1].stop-kindeys[1][1].start,
            linewidth=2,edgecolor='r',facecolor='none'
        )
    )
    ax=plt.subplot2grid((2,4),(1,3),projection="3d")
    p=img_data[kindeys[1]]
    verts, faces ,_,_= measure.marching_cubes(p)
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])


    # plt.savefig("%05d.jpg"%case_id)
    plt.show()
