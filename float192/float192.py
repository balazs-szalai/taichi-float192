# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 20:45:13 2025

@author: balazs
"""
import taichi as ti
from . import mantissa128 as m 

# error flags:
    # 1 << 1: impossible outcome
    # 1 << 2: zero division

f192_t = ti.types.vector(6, ti.u32)

@ti.func
def neg_f192(a: ti.types.vector(6, ti.u32)):
    ret = a
    if a[4] == 0:
        ret[4] = ti.u32(1) | a[4]
    else:
        ret[4] = ti.u32(0) | (a[4] & ti.u32(0xfffffffe))
    return ret

@ti.func
def equalize_exp(v1: ti.types.vector(6, ti.u32), 
                 v2: ti.types.vector(6, ti.u32)):
    ret1 = ti.Vector([0]*6, ti.u32)
    ret2 = ti.Vector([0]*6, ti.u32)
    if v1[5] == v2[5]:
        ret1, ret2 = v1, v2
    elif v1[5] > v2[5]:
        exp = v1[5]
        shift = exp - v2[5]
        mant = m.bit_shift_down_u128(v2, shift)
        
        ret1 = v1
        ret2 = mant
        ret2[4] = v2[4]
        ret2[5] = exp
    else:
        exp = v2[5]
        shift = exp - v1[5]
        mant = m.bit_shift_down_u128(v1, shift)
        
        ret2 = v2
        ret1 = mant
        ret1[4] = v1[4]
        ret1[5] = exp
    return ret1, ret2

@ti.func 
def normalize(a: ti.types.vector(6, ti.u32)):
    shift_limb = m.leading_zero_limbs(a)
    ret = a
    
    if shift_limb < 4:
        shift_bit = 31-m.log2_u32(a[3-shift_limb])
        # print(a[3-shift_limb], log2_u32(a[3-shift_limb]))
        
        if shift_bit != 0:
            ret = m.bit_shift_up_simple(ret, shift_bit)
        
        if shift_limb != 0:
            ret = m.limb_shift_up(ret, shift_limb)
        ret[5] = a[5] - shift_limb*32 - shift_bit 
    
    ret[4] = a[4]
    
    return ret

@ti.func
def add_f192(self, other):
    self, other = equalize_exp(self, other)
    ret = ti.Vector([0]*6, ti.u32)
    
    if self[4]%2 == other[4]%2:
        ret, of = m.add_u128_hi(self, other)
        ret[4] = self[4] | other[4]
        ret[5] = self[5]
        if of:
            ret[5] += 32

    elif self[4]%2 == 1 and other[4]%2 == 0:
        ret = m.sub_u128(other, self)
        
        ret[5] = self[5]
        
        if m.gt_u128(self, other):
            ret[4] = ti.u32(1) | self[4] | other[4]
        else:
            ret[4] = ti.u32(0) | (self[4] & ti.u32(0xfffffffe)) | (other[4] & ti.u32(0xfffffffe))
    
    elif other[4]%2 == 1 and self[4]%2 == 0:
        ret = m.sub_u128(self, other)
        
        ret[5] = self[5]
        
        if m.gt_u128(other, self):
            ret[4] = ti.u32(1) | self[4] | other[4]
        else:
            ret[4] = ti.u32(0) | (self[4] & ti.u32(0xfffffffe)) | (other[4] & ti.u32(0xfffffffe))
    else:
        ret[4] = ti.u32(1 << 1)
    
    
    # print(ret)
    ret = normalize(ret)
    return ret

@ti.func 
def sub_f192(self, other):
    self, other = equalize_exp(self, other)
    # print(self, other)
    ret = ti.Vector([0]*6, ti.u32)
    
    if self[4]%2 != other[4]%2:
        ret, of = m.add_u128_hi(self, other)
        ret[4] = self[4] | (other[4] & ti.u32(0xfffffffe))
        ret[5] = self[5]
        if of:
            ret[5] += 32 
    else:
        ret = m.sub_u128(self, other)
        
        if m.gt_u128(other, self):
            ret[4] = ti.u32(1)
        else:
            ret[4] = ti.u32(0)
        
        ret[4] ^= self[4]%2
        ret[4] |= (self[4] & ti.u32(0xfffffffe)) | (other[4] & ti.u32(0xfffffffe))
        ret[5] = self[5]
    
    ret = normalize(ret)
    return ret

@ti.func 
def mul_f192(self, other):
    ret, of = m.mul_u128_hi(self, other)
    # print(ret, of)
    ret[5] = (self[5] & ti.u32(0x7fffffff)) + (other[5] & ti.u32(0x7fffffff))
    if (self[5] & ti.u32(0x80000000)) == (other[5] & ti.u32(0x80000000)):
        ret[5] ^= ti.u32(0x80000000)
    ret[5] += of
    ret[4] = ti.u32(self[4] != other[4]) | ((self[4] | other[4]) & ti.u32(0xfffffffe))
    
    return ret

@ti.func
def f32_to_f192(f: ti.f32):
    exp = ti.u32(0)
    exp0 = -ti.i32(ti.math.log2((2**24-1)/abs(f)))
    mant = ti.u32(abs(f) / ti.pow(ti.f32(2.0), exp0))
    sgn = ti.u32(f != abs(f))
    
    ret = m.u128()
    ret[0] = mant
    ret[4] = sgn
    if exp0 >= 0:
        exp = ti.u32(exp0)
        ret[5] = ti.u32(0x80000000) | ti.u32(exp)
    else:
        exp = ti.u32(-exp0)
        ret[5] = ti.u32(0x80000000) - exp 
    return normalize(ret)

@ti.func
def f192_to_f32(a: ti.types.vector(6, ti.u32)):
    mant = a[3] >> 8
    exp = ti.i32(0)
    if a[5] > ti.u32(0x80000000):
        exp = ti.i32(a[5] - ti.u32(0x80000000)) + 104
    else:
        exp = -ti.i32(ti.u32(0x80000000) - a[5]) + 104
    sgn = a[4]%2
    
    ret = ti.f32(mant * ti.pow(ti.f32(2.0), exp))
    if sgn:
        ret *= -1
    
    return ret

@ti.func 
def div_f192(self, other):
    zero = ti.Vector([0]*6, ti.u32)
    ret = ti.Vector([0]*6, ti.u32)
    
    if not m.eq_u128(other, zero):
        o_inv = f32_to_f192(1/f192_to_f32(other))
        two = normalize(ti.Vector([2,0,0,0,0,ti.u32(0x80000000)], ti.u32))
        
        ti.loop_config(serialize=True)
        for i in range(7):
            o_inv = mul_f192(o_inv, sub_f192(two, mul_f192(other, o_inv)))
        
        ret = mul_f192(self, o_inv)
    else:
        ret = ti.Vector([ti.u32(0xffffffff)]*6, ti.u32)
        ret[4] = ti.u32(1) if self[4]%2 != other[4]%2 else ti.u32(0)
        ret[4] |= (self[4] & ti.u32(0xfffffffe)) | (other[4] & ti.u32(0xfffffffe))
        ret[4] |= 1 << 2
        ret[5] = ti.u32(0xfffeffff)
    
    return ret
    
#%% test
if __name__ == '__main__':
    ti.init(arch=ti.cpu, fast_math=False, advanced_optimization = False)
    
    @ti.kernel
    def test_f192():
        a = f32_to_f192(-289.80362)
        b = f32_to_f192(-64.00007)
        
        p = div_f192(a, b)
        print(p)
        fp = f192_to_f32(p)
        print(fp)
    
    test_f192()
