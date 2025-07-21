# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 19:02:45 2025

@author: balazs
"""

import torch
import taichi as ti
import float192 as f192ops
import numpy as np
import matplotlib.pyplot as plt

ti.init(arch=ti.cpu, advanced_optimization = False)  # or ti.cuda
vec_u32_6 = ti.types.vector(6, ti.u32)

M, N = 5000, 5000  # test grid size

a = (torch.rand((M, N), dtype=torch.float32)-0.5)*1000
b = (torch.rand((M, N), dtype=torch.float32)-0.5)*1000

a_data = np.empty((M, N, 6), dtype = np.uint32)
b_data = np.empty((M, N, 6), dtype = np.uint32)

# Result arrays (in f192 format and then f32)
add_res = torch.empty((M, N), dtype=torch.float32)
sub_res = torch.empty((M, N), dtype=torch.float32)
mul_res = torch.empty((M, N), dtype=torch.float32)
div_res = torch.empty((M, N), dtype=torch.float32)


@ti.kernel 
def test(a: ti.types.ndarray(ti.f32, 2),
         b: ti.types.ndarray(ti.f32, 2),
         a_tmp: ti.types.ndarray(vec_u32_6, 2),
         b_tmp: ti.types.ndarray(vec_u32_6, 2),
         add_res: ti.types.ndarray(ti.f32, 2),
         sub_res: ti.types.ndarray(ti.f32, 2),
         mul_res: ti.types.ndarray(ti.f32, 2),
         div_res: ti.types.ndarray(ti.f32, 2)):
    m, n = a.shape
    
    for i, j in ti.ndrange(m, n):
        a_tmp[i, j] = f192ops.f32_to_f192(a[i, j])
        b_tmp[i, j] = f192ops.f32_to_f192(b[i, j])
    
    for i, j  in ti.ndrange(m, n):
        add_res[i, j] = f192ops.f192_to_f32(f192ops.add_f192(a_tmp[i, j], b_tmp[i, j]))
        sub_res[i, j] = f192ops.f192_to_f32(f192ops.sub_f192(a_tmp[i, j], b_tmp[i, j]))
        mul_res[i, j] = f192ops.f192_to_f32(f192ops.mul_f192(a_tmp[i, j], b_tmp[i, j]))
        div_res[i, j] = f192ops.f192_to_f32(f192ops.div_f192(a_tmp[i, j], b_tmp[i, j]))

test(a, b, a_data, b_data, add_res, sub_res, mul_res, div_res)

#%%
add_err = torch.sum(a+b-add_res)
sub_err = torch.sum(a-b-sub_res)
mul_err = torch.sum(a*b-mul_res)
div_err = np.sum((a/b-div_res).numpy()[~np.isnan((a/b-div_res).numpy())])

#%%
print(add_err, sub_err, mul_err, div_err)

plt.close('all')
fig, ax = plt.subplots()
div_errs = (a/b-div_res).numpy()
div_errs[np.isnan(div_errs)] = 1
plt.imshow(np.log(np.abs(div_errs+1e-15)))

# fig, ax = plt.subplots()
# plt.imshow(a.numpy() > 0)

# fig, ax = plt.subplots()
# plt.imshow(b.numpy() > 0)

# fig, ax = plt.subplots()
# plt.imshow((a*b).numpy() < 0)