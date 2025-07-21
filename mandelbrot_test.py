# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 15:03:32 2025

@author: balazs
"""

import taichi as ti
from float192 import supports_f192, f192_t, str_to_f192, f192_to_f32, normalize
from float192 import i32_to_f192 as f192
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ti.init(arch=ti.cpu)
    
    @ti.func
    @supports_f192(globals(), verbose=True)
    def complex_sqr_x(x: f192_t, y: f192_t) -> f192_t:
        return x*x - y*y#ti.Vector([z[0] * z[0] - z[1] * z[1], 2 * z[0] * z[1]], ti.float64)
    
    @ti.func
    @supports_f192(globals(), verbose=True)
    def complex_sqr_y(x: f192_t, y: f192_t) -> f192_t:
        two = f192(2)#ti.Vector([0, 0, 0, ti.u32(2147483648), 0, ti.u32(2147483522)], ti.u32)
        return two * x*y#ti.Vector([z[0] * z[0] - z[1] * z[1], 2 * z[0] * z[1]], ti.float64)
    
    @ti.func
    @supports_f192(globals(), verbose=True)
    def complex_sqr(x: f192_t, y: f192_t) -> [f192_t, f192_t]:
        two = f192(2)
        return [x*x - y*y, two * x*y]#ti.Vector([z[0] * z[0] - z[1] * z[1], 2 * z[0] * z[1]], ti.float64)
    
    @ti.kernel
    @supports_f192(globals(), verbose=True)
    def compute(x0: f192_t,
                y0: f192_t,
                x1: f192_t,
                y1: f192_t,
                m0: ti.int32,
                n0: ti.int32,
                img: ti.types.ndarray(ti.f32, 2),
                iter_depth: ti.int32):
        x = x1-x0
        y = y1-y0
        
        m, n = m0, n0
        
        if img.shape[0] < m0 or img.shape[1] < n0:
            m, n = img.shape
        
        for i, j in ti.ndrange(m, n):
            cx = x0 + f192(j)/f192(n)*x
            cy = y0 + f192(i)/f192(m)*y
            iterations = 0
            zx = f192(0)
            zy = f192(0)
            
            while zx*zx + zy*zy < f192(4) and iterations < iter_depth:
                zx_old, zy_old = zx, zy
                zx = complex_sqr_x(zx_old, zy_old) + cx
                zy = complex_sqr_y(zx_old, zy_old) + cy
                iterations += 1
                # print(f192_to_f32(zx*zx + zy*zy))
            val = iterations/iter_depth
            # print(iterations)
            img[i, j] = val
    
    #%%
    x0 = str_to_f192('-0.17032344376207073')
    y0 = str_to_f192('-1.0402289976592387')
    x1 = str_to_f192('-0.1703234437620603')
    y1 = str_to_f192('-1.0402289976592325')
    m, n = 400, 600
    iter_depth = 256
    img = np.zeros((m, n), dtype = np.float32)
    
    #%%
    compute(x0, y0, x1, y1, m, n, img, iter_depth)
    
    plt.imshow(img)
    
        
        
