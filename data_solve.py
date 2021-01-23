import numpy as np
import trimesh
import xlrd

# 打开文件(以xlsx存储在标准模型人工标注的穴位)
data = xlrd.open_workbook('E:/SeniorYearUp/Final/Material/Wechat/穴位表.xlsx')
table = data.sheet_by_name('Sheet1')
Xuewei_num = table.nrows - 1
print("已标记穴位数：" + str(Xuewei_num))


#导入标准模型
mesh_test = trimesh.load('E:\\SeniorYearUp\\Final\\project\\Output\\meshes\\standard\\000.obj')
Vertices = np.asarray(mesh_test.vertices)
print(Vertices.shape)

#计算映射矩阵
K = np.zeros([Xuewei_num,3])
for idx in range(Xuewei_num):
    s = table.cell(idx+1,1).value
    x_abs = -float(s[:-1])
    s = table.cell(idx+1,2).value
    y_abs = float(s[:-1])
    s = table.cell(idx+1,3).value
    z_abs = float(s[:-1])
    pos_support = np.zeros([3,3])
    for i in range(3):
        pos_idx = int(table.cell(idx+1,4+i).value)
        pos_support[:,i] = Vertices[pos_idx,:].T
    K[idx,:] = np.dot(np.linalg.inv(pos_support),np.asarray([x_abs,y_abs,z_abs]).T).T   

#以嵌套字典的形式存储相应支撑点
Xuewei = {}
for idx in range(Xuewei_num):
    tmp = Xuewei.setdefault(table.cell(1+idx,0).value, {})
    tmp2 = tmp.setdefault('x_abs', table.cell(1+idx,1).value)
    tmp2 = tmp.setdefault('y_abs', table.cell(1+idx,2).value)
    tmp2 = tmp.setdefault('z_abs', table.cell(1+idx,3).value)
    tmp2 = tmp.setdefault('point_support_0', table.cell(1+idx,4).value)
    tmp2 = tmp.setdefault('point_support_1', table.cell(1+idx,5).value)
    tmp2 = tmp.setdefault('point_support_2', table.cell(1+idx,6).value)
    tmp2 = tmp.setdefault('k0', K[idx,0])
    tmp2 = tmp.setdefault('k1', K[idx,1])
    tmp2 = tmp.setdefault('k2', K[idx,2])

np.save('E:\\SeniorYearUp\\Final\\project\\Output\\meshes\\Xuewei.npy', Xuewei)