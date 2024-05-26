import os
import numpy as np
import torch 
from PIL import Image
def ravel_hash_vec(postion):
    """
    Ravel the coordinates after subtracting the min coordinates.
    """
    assert postion.ndim == 2
    postion = postion.clone()
    postion -= postion.amin(0)
    postion = postion.type(torch.long)
    postion_max = postion.amax(0).type(torch.long) + 1

    keys = torch.zeros(postion.shape[0], dtype=torch.long)
    # Fortran style indexing
    for j in range(postion.shape[1] - 1):
        keys += postion[:, j]
        keys *= postion_max[j + 1]
    keys += postion[:, -1]
    return keys

def voxelsize(coord, voxel_size=0.01, hash_type='ravel'):
    # 扩张
    postion = coord / torch.tensor([voxel_size,voxel_size,1])
    postion[:,:2] = torch.floor(postion[:,:2])  # 划分体素框，立方柱
    
    key = ravel_hash_vec(postion)
    idx_sort = torch.argsort(key)
    key_sort = key[idx_sort]
    _, count = torch.unique(key_sort, return_counts=True)
    z_sort = postion[idx_sort,2]
    max_part_idxs = []
    start = 0
    # 筛选z轴最高点索引
    for c in count:
        end = start+c
        max_part_idx = z_sort[start:end].argmax(axis = 0)
        max_part_idxs.append(max_part_idx)
        start = end
    max_part_idxs = torch.stack(max_part_idxs)
    idx_select = torch.tensor(np.insert(count, 0, 0)[0:-1]).cumsum(dim=0) + max_part_idxs % count  
    # 获得体素化后的原数据索引
    idx_unique = idx_sort[idx_select] 

    return idx_unique,postion[idx_unique]

# def one_to_many():
    

if __name__ == '__main__':
    # filepath = "./data/points.txt"
    # filepath = "./data/scene0640_00.txt"
    filepath = "Area_1_1 - Cloud.txt"
    voxel_size = 1
    patch_edge= 0 # 空白块填充
    extend_size = 0 # 数据扩张维度
    # todo: 手动设定图像大小，通过point的max与min来选择根据h或w计算voxelsize大小，同时可定位到画布中心
    # image_size = None # 若imagesize非None，优先采用imagesize作为图像大小的设定而不是voxelsize，default：(hw)

    points = np.loadtxt(filepath)[:,:6]
    print("data length",points.shape[0])
    xyz,rgb = points[:,:3],points[:,3:6]
    xyz = xyz-xyz.min(axis = 0)
    # 体素化，full_pixel：填充像素坐标；idx_unique：像素坐标对应点云的rgb

    # >>>>>>数据扩充1>>>>>>
    # if extend_size > 0: 
    #     ext_xyz_list = []
    #     ext_rgb_list = []
    #     for e1 in range(-1,extend_size+1):
    #         for e2 in range(-1,extend_size+1):            
    #             ext_xyz_list.append(np.stack([xyz[:,0]+voxel_size*e2,xyz[:,1]+voxel_size*e1,xyz[:,2]+voxel_size],axis=1))
    #             ext_rgb_list.append(rgb)
    #     xyz = np.concatenate(ext_xyz_list,axis=0) # 重构的数据
    #     rgb = np.concatenate(ext_rgb_list,axis=0)
    # >>>>>>>>>>>>>>>>>>
    idx_unique, full_pixel = voxelsize(torch.from_numpy(xyz),voxel_size)
    idx_unique= idx_unique.numpy()
    full_pixel = full_pixel.numpy()
    rgb = rgb[idx_unique]# 取出对应点的rgb

    # >>>>>>数据扩充2>>>>>>
    if extend_size > 0: 
        ext_xyz_list = []
        ext_rgb_list = []
        for e1 in range(-1,extend_size+1):
            for e2 in range(-1,extend_size+1):            
                ext_xyz_list.append(np.stack([full_pixel[:,0]+e1,full_pixel[:,1]+e2,full_pixel[:,2]],axis=1))
                ext_rgb_list.append(rgb)
        full_pixel = np.concatenate(ext_xyz_list,axis=0) # 重构的数据
        rgb = np.concatenate(ext_rgb_list,axis=0)
        # 重复点消除
        idx_unique, full_pixel = voxelsize(torch.from_numpy(full_pixel),voxel_size=1)# full_pixel 为的距离为1
        idx_unique= idx_unique.numpy()
        full_pixel = full_pixel.numpy()
        rgb = rgb[idx_unique]
    # >>>>>>>>>>>>>>>>>>
    pixel_z=full_pixel[:,2] # 单拿出z
    full_pixel=full_pixel[:,:2].astype (np.int32)
    print("像素数目",full_pixel.shape)
    fig_max = full_pixel.max(axis=0)
    fig_min = full_pixel.min(axis=0)
    fig_whc = (int(fig_max[0]-fig_min[0]+1+2*patch_edge),int(fig_max[1]-fig_min[1]+1+2*patch_edge),3) # 画布长度
    fig_whz = (int(fig_max[0]-fig_min[0]+1+2*patch_edge),int(fig_max[1]-fig_min[1]+1+2*patch_edge)) # 画布长度
    print("填充点数量",full_pixel.shape)
    im = np.full(fig_whc,255) # 申请白画布
    im_z = np.full(fig_whz,0.) # 申请画布高度，便于消除白色区域
    print("画布大小",im.shape)
    im[full_pixel[:,0]+patch_edge,full_pixel[:,1]+patch_edge] = rgb # 渲染
    im_z[full_pixel[:,0]+patch_edge,full_pixel[:,1]+patch_edge] = pixel_z # 渲染


    # >>>>>>消除白色区域>>>>>>
    patch_rgb = []
    patch_z = []
    if patch_edge >0 :
        # 取出白色区域
        # x,y = np.where(np.sum(im == 255,axis = 2)==3) # 空白区域的坐标
        # xy避免在最边缘
        x,y = np.where(np.sum(im[patch_edge:-patch_edge,patch_edge:-patch_edge] == 255,axis = 2)==3) # 空白区域的坐标
        # print(x.shape[0])
        i = 0
        for eh in range(-patch_edge,patch_edge+1):
            for ew in range(-patch_edge,patch_edge+1):
                patch_rgb.append(im[x+eh,y+ew]) # 获得rgb
                patch_z.append(im_z[x+eh,y+ew]) # 获得对应像素高度
                i+=1
        patch_rgb = np.stack(patch_rgb,axis = 1)# [m,patch,3]
        patch_z = np.stack(patch_z,axis = 1)# [m,patch,1]
        # 排序
        print("patch中像素数量",patch_rgb.shape[1])
        n_indices = np.arange(patch_rgb.shape[0]).reshape(patch_rgb.shape[0],1).repeat(patch_rgb.shape[1],axis =1) 
        idx = patch_z.argmax(axis=1) # 根据高度从小到大排序
        idx = idx.reshape(patch_rgb.shape[0],1) 
        # np.where(patch_z[1000].sum(axis=1)>0)
        rgb = patch_rgb[n_indices,idx,:][:,0,:]
        im[patch_edge:-patch_edge,patch_edge:-patch_edge][x,y] = rgb # [m,3]
    # >>>>>>>>>>>>>>>>>>
    im = Image.fromarray(im.astype('uint8')).convert('RGB')
    # im.show()
    im.save("./images"+os.path.basename(filepath).split(".")[0]+"_vl_{} es_{} pe_{}".format(voxel_size,extend_size,patch_edge)+".png")