# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 20:45:13 2025

@author: balazs
"""
import taichi as ti
from . import mantissa128 as m 
from mpmath import mp, mpf

mp.prec = 200


# error flags:
    # 1 << 1: impossible outcome
    # 1 << 2: zero division
    # 1 << 3: unhandled case of f32 underflow in division
    

f192_t = ti.types.vector(6, ti.u32)

@ti.real_func
def neg_f192(a: ti.types.vector(6, ti.u32)) -> f192_t:
    ret = a
    if a[4] == 0:
        ret[4] = ti.u32(1) | a[4]
    else:
        ret[4] = ti.u32(0) | (a[4] & ti.u32(0xfffffffe))
    return ret

@ti.real_func
def equalize_exp(v10: ti.types.vector(6, ti.u32), 
                 v20: ti.types.vector(6, ti.u32)) -> [f192_t, f192_t]:
    ret1 = ti.Vector([0]*6, ti.u32)
    ret2 = ti.Vector([0]*6, ti.u32)
    
    v1, v2 = v10, v20
    
    cond1 = m.eq_u128(v10, ret1)
    cond2 = m.eq_u128(v20, ret1)
    
    if cond1 and cond2:
        v1[5] = ti.u32(0x80000000)
        v2[5] = ti.u32(0x80000000)
    elif cond1:
        v1[5] = 0
    elif cond2:
        v2[5] = 0
    
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
    return [ret1, ret2]

@ti.real_func 
def normalize(a: ti.types.vector(6, ti.u32)) -> f192_t:
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

@ti.real_func
def add_f192(self0: f192_t, other0: f192_t) -> f192_t:
    self, other = equalize_exp(self0, other0)
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

@ti.real_func 
def sub_f192(self0: f192_t, other0: f192_t) -> f192_t:
    self, other = equalize_exp(self0, other0)
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
        
        # print(ret)
        # print(normalize(ret))
    
    ret = normalize(ret)
    return ret

@ti.real_func 
def mul_f192(self: f192_t, other: f192_t) -> f192_t:
    ret, of = m.mul_u128_hi(self, other)
    # print(ret, of)
    ret[5] = (self[5] & ti.u32(0x7fffffff)) + (other[5] & ti.u32(0x7fffffff))
    if (self[5] & ti.u32(0x80000000)) == (other[5] & ti.u32(0x80000000)):
        ret[5] ^= ti.u32(0x80000000)
    ret[5] += of
    ret[4] = ti.u32(self[4] != other[4]) | ((self[4] | other[4]) & ti.u32(0xfffffffe))
    
    return ret

@ti.real_func
def f32_to_f192(f: ti.f32) -> f192_t:
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

@ti.real_func
def f192_to_f32(a: ti.types.vector(6, ti.u32)) -> ti.f32:
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

@ti.real_func 
def div_f192(self:f192_t, other:f192_t) -> f192_t:
    zero = ti.Vector([0]*6, ti.u32)
    ret = ti.Vector([0]*6, ti.u32)
    # tmp1 = ti.Vector([0]*6, ti.u32)
    # tmp2 = ti.Vector([0]*6, ti.u32)
    # tmp3 = ti.Vector([0]*6, ti.u32)
    
    if not m.eq_u128(other, zero):
        o_f32 = f192_to_f32(other)
        o_inv = normalize(ti.Vector([0,0,0,0,0,ti.u32(0x80000000)], ti.u32))
        if o_f32 != 0:
            o_inv = f32_to_f192(1/f192_to_f32(other))
        else:
            o_inv[4] |= 1 << 3
        two = normalize(ti.Vector([2,0,0,0,0,ti.u32(0x80000000)], ti.u32))
        # print(f192_to_f32(two))
        # print(two)
        
        ti.loop_config(serialize=True)
        for i in range(7):
            # print(f192_to_f32(o_inv), f192_to_f32(mul_f192(other, o_inv)), f192_to_f32(sub_f192(two, mul_f192(other, o_inv))), f192_to_f32(mul_f192(o_inv, sub_f192(two, mul_f192(other, o_inv)))))
            o_inv = mul_f192(o_inv, sub_f192(two, mul_f192(other, o_inv)))
        
        ret = mul_f192(self, o_inv)
    else:
        ret = ti.Vector([ti.u32(0xffffffff)]*6, ti.u32)
        ret[4] = ti.u32(1) if self[4]%2 != other[4]%2 else ti.u32(0)
        ret[4] |= (self[4] & ti.u32(0xfffffffe)) | (other[4] & ti.u32(0xfffffffe))
        ret[4] |= 1 << 2
        ret[5] = ti.u32(0xfffeffff)
    
    return ret

@ti.real_func 
def cmp_f192(self: f192_t, other: f192_t) -> ti.i32:
    v1, v2 = equalize_exp(self, other)
    return m.cmp_u128(v1, v2)

@ti.real_func 
def gt_f192(self: f192_t, other: f192_t) -> ti.i32:
    return cmp_f192(self, other) == 1
@ti.real_func 
def eq_f192(self: f192_t, other: f192_t) -> ti.i32:
    return cmp_f192(self, other) == 0
@ti.real_func 
def lt_f192(self: f192_t, other: f192_t) -> ti.i32:
    return cmp_f192(self, other) == -1
@ti.real_func 
def ge_f192(self: f192_t, other: f192_t) -> ti.i32:
    v = cmp_f192(self, other) 
    return v == 1 or v == 0
@ti.real_func 
def le_f192(self: f192_t, other: f192_t) -> ti.i32:
    v = cmp_f192(self, other) 
    return v == -1 or v == 0

@ti.real_func 
def i32_to_f192(val: ti.i32) -> f192_t:
    ret = ti.Vector([0]*6, ti.u32)
    ret[0] = ti.u32(abs(val))
    if val < 0:
        ret[4] = ti.u32(1)
    else:
        ret[4] = ti.u32(0)
    ret[5] = ti.u32(0x80000000)
    return normalize(ret)

def str_to_f192(val: str):
    x = mpf(val)
    sign = 0 if x >= 0 else 1
    x = abs(x)
    
    m, e = mp.frexp(x)
    
    mantissa = int(m * (1 << 128))
    mantissa_u32 = [(mantissa >> (32 * i)) & 0xFFFFFFFF for i in range(4)]
    mantissa_u32.append(sign)
    mantissa_u32.append(0x80000000 + e - 128)
    

    return ti.Vector(mantissa_u32, ti.u32)
    
#%% test
if __name__ == '__main__':
    ti.init(arch=ti.cpu, fast_math=False, advanced_optimization = False)
    
    @ti.kernel
    def test_f192():
        two = normalize(ti.Vector([2,0,0,0,0,ti.u32(0x80000000)], ti.u32))
        print(two)
        # one = ti.Vector([0, 0, ti.u32(2021654528), ti.u32(4294967182), 0, ti.u32(2147483520)], ti.u32)
        # two, one = equalize_exp(two, one)
        # print(two, one, m.neg_u128(one))
        # print(f192_to_f32(sub_f192(two, one)))
        # a = f32_to_f192(-289.80362)
        # b = f32_to_f192(-64.00007)
        # # o_inv = ti.Vector([ti.u32(4294966532), ti.u32(2617298183), 10616820, ti.u32(2147481344), 1, ti.u32(2147483515)], ti.u32)
        # # print(b, o_inv, f192_to_f32(mul_f192(b, o_inv)))
        # # print(mul_f192(b, o_inv))
        # # print(m.mul_full_u128(b, o_inv))
        
        # # s = sub_f192(a, b)
        # p = div_f192(a, b)
        # # print(s)
        # print(p)
        
        # # fs = f192_to_f32(s)
        # fp = f192_to_f32(p)
        
        # # print(fs)  # Expected ≈ -0.75
        # print(fp)  # Expected ≈ -3.375
        
        # Optional: Check error flags
        # print("Add flags:", s[4])
        # print("Mul flags:", p[4])
    
    test_f192()
