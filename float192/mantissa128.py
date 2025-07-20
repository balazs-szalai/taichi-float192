# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 20:27:53 2025

@author: balazs
"""

import taichi as ti

class U128:
    pass

@ti.func 
def add_with_carry(a: ti.u32,
                   b: ti.u32,
                   carry_in: ti.u32):
    temp = ti.u32(a+b)
    result = temp + carry_in
    carry_out = ti.u32((temp < a) or (result < temp))
    
    return result, carry_out

@ti.func 
def add_full_u128(a: ti.types.vector(6, ti.u32),
                        b: ti.types.vector(6, ti.u32)):
    
    result = ti.Vector([0]*6, ti.u32)
    
    carry = ti.u32(0)
    ti.loop_config(serialize=True)
    for i in range(4):
        x = a[i]
        y = b[i]
        sum_, new_carry = add_with_carry(x, y, carry)
        result[i] = sum_
        carry = new_carry
    
    return result, carry

@ti.func 
def add_u128_hi(a: ti.types.vector(6, ti.u32),
                b: ti.types.vector(6, ti.u32)):
    result = ti.Vector([0]*6, ti.u32)
    overflow = False
    
    res, carry = add_full_u128(a, b)
    
    if carry:
        overflow = True
        for i in range(3):
            result[i] = res[i+1]
        result[3] = carry 
    else:
        for i in range(4):
            result[i] = res[i]
    
    return result, overflow


@ti.func 
def neg_u128(a: ti.types.vector(6, ti.u32)):
    result = ti.Vector([0]*6, ti.u32)
    
    for i in range(4):
        result[i] = ti.u32(0xffffffff)-a[i]
    # print(result)
    k = 0
    carry = ti.u32(1)
    while carry:
        temp = result[k]
        # print(temp, carry)
        temp, carry = add_with_carry(temp, carry, 0)
        # print(temp, carry)
        result[k] = temp
        k += 1
    return result

@ti.func 
def sub_u128(a: ti.types.vector(6, ti.u32),
             b: ti.types.vector(6, ti.u32)):
    
    res, carry  = add_full_u128(a, neg_u128(b))
    ret = res
    
    if not carry:
        ret = neg_u128(res)
    return ret

@ti.func 
def from_u32_to_u16(a: ti.types.vector(6, ti.u32)):
    ret = ti.Vector([0]*8, ti.u16)
    
    for i in range(4):
        hi = ti.u16(a[i] >> 16)
        lo = ti.u16(a[i] & 0xffff)
        
        ret[2*i] = lo
        ret[2*i + 1] = hi
    
    return ret

@ti.func 
def from_u16_to_u32(a: ti.types.vector(8, ti.u16)):
    ret = ti.Vector([0]*6, ti.u32)
    
    for i in range(4):
        hi = ti.u32(a[2*i+1])
        lo = ti.u32(a[2*i])
        
        ret[i] = (hi << 16) | lo
    
    return ret

@ti.func 
def mul_u128_u16impl(a: ti.types.vector(8, ti.u16),
                     b: ti.types.vector(8, ti.u16)):
    result = ti.Vector([0]*16, ti.u16)
    hi = ti.Vector([0]*8, ti.u16)
    lo = ti.Vector([0]*8, ti.u16)
    
    # ti.loop_config(serialize=True)
    for i in range(8):
        # ti.loop_config(serialize=True)
        for j in range(8):
            tmp = ti.u32(ti.u32(a[i])*ti.u32(b[j]))
            high = tmp >> 16
            low = tmp & 0xffff
            
            temp = ti.u32(result[i+j]) + low
            carry = temp >> 16
            temp &= 0xffff
            result[i+j] = ti.u16(temp)
            
            temp = ti.u32(result[i+j+1]) + high + carry
            carry = temp >> 16
            temp &= 0xffff
            result[i+j+1] = ti.u16(temp)
            
            k = 2
            while carry:
                temp = ti.u32(result[i+j+k]) + carry
                carry = temp >> 16
                temp &= 0xffff
                result[i+j+k] = ti.u16(temp)
                k += 1
    
    for i in range(8):
        hi[i] = result[i+8]
        lo[i] = result[i]
    
    return hi, lo

@ti.func 
def mul_full_u128(a: ti.types.vector(6, ti.u32),
                  b: ti.types.vector(6, ti.u32)):
    
    a16 = from_u32_to_u16(a)
    b16 = from_u32_to_u16(b)
    
    hi16, lo16 = mul_u128_u16impl(a16, b16)
    
    hi = from_u16_to_u32(hi16)
    lo = from_u16_to_u32(lo16)
    
    return hi, lo

@ti.func 
def mul_u128_lo(a: ti.types.vector(6, ti.u32),
                b: ti.types.vector(6, ti.u32)):
    
    return mul_full_u128(a, b)[1]

@ti.func 
def leading_zero_limbs(a: ti.types.vector(6, ti.u32)) -> int:
    ret = 0
    flag = True 
    
    # ti.loop_config(serialize=True)
    for j in range(4):
        i = 3-j
        if a[i] == 0 and flag:
            ret += 1
        else:
            flag = False
    return ret

@ti.func
def log2_u32(x: ti.u32) -> ti.i32:
    low = 0
    high = 31
    
    while low <= high:
        mid = (low + high) // 2
        if x >> mid == 0:
            high = mid - 1
        else:
            low = mid + 1
    return high

@ti.func 
def mul_u128_hi(a: ti.types.vector(6, ti.u32),
             b: ti.types.vector(6, ti.u32)):
    hi, lo = mul_full_u128(a, b)
    # print(hi, lo)
    shift_limb = leading_zero_limbs(hi)
    shift_bit = 0
    
    if shift_limb < 4:
        shift_bit = 31-log2_u32(hi[3-shift_limb])
        # print(shift_limb, shift_bit)
        
        # print(hi)
        hi = bit_shift_up_simple(hi, shift_bit)
        # print(hi)
        hi[0] |= lo[3] >> (32-shift_bit)
        # lo = bit_shift_up_simple(lo, shift_bit)
        
        hi = limb_shift_up(hi, shift_limb)
        # print(hi)
        for i in range(shift_limb):
            hi[i] = lo[4-shift_limb+i]
    else:
        shift_limb -= 4
        shift_bit = 31-log2_u32(hi[3-shift_limb])
        
        hi = bit_shift_up_simple(lo, shift_bit)
        hi = limb_shift_up(hi, shift_limb)
    
    return hi, 128-shift_limb*32-shift_bit

@ti.func 
def bit_shift_up_simple(a: ti.types.vector(6, ti.u32), shift: int):
    result = ti.Vector([0]*6, ti.u32)
    if shift > 0:
        result = ti.Vector([0]*6, ti.u32)
        high, low = ti.u32(0), ti.u32(0)
        
        ti.loop_config(serialize=True)
        for i in range(4):
            low = (a[i] << shift) | high
            high = a[i] >> (32-shift)
            result[i] = low
    else:
        result = ti.Vector([a[i] for i in range(6)], ti.u32)
    return result

@ti.func 
def limb_shift_up(a: ti.types.vector(6, ti.u32), n: int):
    result = ti.Vector([0]*6, ti.u32)
    for i in range(4-n):
        result[i+n] = a[i]
    return result

@ti.func
def bit_shift_up_u128(a: ti.types.vector(6, ti.u32), shift: int):
    n = shift//32 
    shift %= 32 
    result = ti.Vector([a[i] for i in range(4)])
    if shift:
        result = bit_shift_up_simple(a, shift)
    
    if n:
        result = limb_shift_up(result, n)
    
    return  result


@ti.func 
def bit_shift_down_simple(a: ti.types.vector(6, ti.u32), shift: int):
    result = ti.Vector([0]*6, ti.u32)
    high, low = ti.u32(0), ti.u32(0)
    
    # ti.loop_config(serialize=True)
    for j in range(4):
        i = 3-j
        
        low = (a[i] >> shift) | high
        high = a[i] << (32-shift)
        
        result[i] = low
    return result

@ti.func 
def limb_shift_down(a: ti.types.vector(6, ti.u32), n: int):
    result = ti.Vector([0]*6, ti.u32)
    for j in range(4-n):
        i = 3-j
        
        result[i-n] = a[i]
    return result

@ti.func
def bit_shift_down_u128(a: ti.types.vector(6, ti.u32), shift: int):
    n = shift//32 
    shift %= 32 
    
    result = ti.Vector([a[i] for i in range(6)])
    if shift:
        result = bit_shift_down_simple(a, shift)
    
    if n:
        result = limb_shift_down(result, n)
    
    return  result

@ti.func
def cmp_u128(a: ti.types.vector(6, ti.u32), b: ti.types.vector(6, ti.u32)) -> ti.i32:
    ret = 0
    
    for j in range(4):
        i = 3-j
        if not ret:
            if a[i] < b[i]:
                ret = -1
            elif a[i] > b[i]:
                ret = 1
    return ret

@ti.func
def lt_u128(a: ti.types.vector(6, ti.u32), b: ti.types.vector(6, ti.u32)):
    return cmp_u128(a, b) == -1
@ti.func
def gt_u128(a: ti.types.vector(6, ti.u32), b: ti.types.vector(6, ti.u32)):
    return cmp_u128(a, b) == 1
@ti.func
def eq_u128(a: ti.types.vector(6, ti.u32), b: ti.types.vector(6, ti.u32)):
    return cmp_u128(a, b) == 0

@ti.func 
def u128():
    return ti.Vector([0]*6, ti.u32)

@ti.func 
def leading_zero_limbs_u16impl(a: ti.types.vector(10, ti.u16)) -> int:
    ret = 0
    flag = True 
    # ti.loop_config(serialize=True)
    for j in range(10):
        i = 9-j
        if a[i] == 0 and flag:
            ret += 1
        else:
            flag = False
    return ret

@ti.func 
def bit_shift_up_simple_u16impl(a: ti.types.vector(10, ti.u16), shift: int):
    result = ti.Vector([0]*10, ti.u16)
    high, low = ti.u16(0), ti.u16(0)
    
    # ti.loop_config(serialize=True)
    for i in range(10):
        low = (a[i] << shift) | high
        high = a[i] >> (16-shift)
        result[i] = low
    return result

@ti.func 
def bit_shift_down_simple_u16impl(a: ti.types.vector(10, ti.u16), shift: int):
    result = ti.Vector([0]*10, ti.u16)
    high, low = ti.u16(0), ti.u16(0)
    
    # ti.loop_config(serialize=True)
    for j in range(10):
        i = 9-j
        
        low = (a[i] >> shift) | high
        high = a[i] << (16-shift)
        
        result[i] = low
    return result

@ti.func
def cmp_u128_u16impl(a: ti.types.vector(10, ti.u16), b: ti.types.vector(10, ti.u16)) -> ti.i32:
    ret = 0
    
    # ti.loop_config(serialize=True)
    for j in range(10):
        i = 9-j
        if not ret:
            if a[i] < b[i]:
                ret = -1
            elif a[i] > b[i]:
                ret = 1
    return ret

@ti.func 
def mul_u128_u16impl_82(a: ti.types.vector(10, ti.u16),
                        b: ti.types.vector(10, ti.u16)):
    result = ti.Vector([0]*10, ti.u16)
    
    # ti.loop_config(serialize=True)
    for i in range(8):
        # ti.loop_config(serialize=True)
        for j in range(2):
            tmp = ti.u32(ti.u32(a[i])*ti.u32(b[j]))
            high = tmp >> 16
            low = tmp & 0xffff
            
            temp = ti.u32(result[i+j]) + low
            carry = temp >> 16
            temp &= 0xffff
            result[i+j] = ti.u16(temp)
            
            temp = ti.u32(result[i+j+1]) + high + carry
            carry = temp >> 16
            temp &= 0xffff
            result[i+j+1] = ti.u16(temp)
    
    return result

@ti.func 
def extend_by_digit(q: ti.types.vector(10, ti.u16),
                    q_hat: ti.types.vector(10, ti.u16)):
    ret = ti.Vector([0]*10, ti.u16)
    ret[0] = q_hat[0]
    
    carry = ti.u32(q_hat[1])
    # ti.loop_config(serialize=True)
    for i in range(9):
        tmp = ti.u32(q[i]) + carry
        tmp_hi = tmp >> 16
        tmp_lo = tmp & 0xffff
        
        ret[i+1] = ti.u16(tmp_lo)
        carry = tmp_hi
    
    return ret

@ti.func 
def sub_u128_u16impl(a: ti.types.vector(10, ti.u16), b: ti.types.vector(10, ti.u16)):
    ret = ti.Vector([0]*10, ti.u16)
    
    carry = 0
    # ti.loop_config(serialize=True)
    for i in range(10):
        tmp = ti.i32(a[i]) - carry - ti.i32(b[i])
        
        if tmp < 0:
            carry = 1
            ret[i] = ti.u16(2**16+tmp)
        else:
            carry = 0 
            ret[i] = ti.u16(tmp)
    
    return ret

@ti.func
def divmod_u128(a: ti.types.vector(6, ti.u32), 
                b: ti.types.vector(6, ti.u32)):
    zero_devision = True
    for i in range(4):
        if b[i]:
            zero_devision = False
    
    if not zero_devision:
        a_norm = ti.Vector([0]*10, ti.u16)
        b_norm = ti.Vector([0]*10, ti.u16)
        
        a_tmp = from_u32_to_u16(a)
        b_tmp = from_u32_to_u16(b)
        
        for i in range(8):
            a_norm[i] = a_tmp[i]
            b_norm[i] = b_tmp[i]
        
        one = ti.Vector([0]*8, ti.u16)
        one[0] = ti.u16(1)
        shift = leading_zero_limbs_u16impl(b_norm)
        back_shift = False
        
        b_hat = ti.u32(b_norm[9-shift])
        
        norm = 0
        if b_hat < 2**15:
            norm = 15 - log2_u32(b_hat)
            a_norm = bit_shift_up_simple_u16impl(a_norm, norm)
            b_norm = bit_shift_up_simple_u16impl(b_norm, norm)
            back_shift = True
        shift = leading_zero_limbs_u16impl(b_norm)
        shift_a = min(leading_zero_limbs_u16impl(a_norm), shift-1)
        b_hat = ti.u32(b_norm[9-shift])
        
        q = ti.Vector([0]*10, ti.u16)
        q_hat_vec = ti.Vector([0]*10, ti.u16)
        r = ti.Vector([0]*10, ti.u16)
        dig_rem = ti.Vector([0]*10, ti.u16)
        
        for i in range(10-shift+shift_a+1):
            r[i] = a_norm[i+shift-shift_a-1]
        for i in range(shift-shift_a-1):
            dig_rem[i] = a_norm[i]
        
        i = 0
        while True:            
            a_hat = ti.u32(r[10-shift])*2**16 + ti.u32(r[9-shift])
            
            q_hat = a_hat//b_hat
            q_hat_hi = q_hat >> 16
            q_hat_lo = q_hat & 0xffff
            q_hat_vec[0] = ti.u16(q_hat_lo)
            q_hat_vec[1] = ti.u16(q_hat_hi)
            
            count = 0
            tmp = mul_u128_u16impl_82(b_norm, q_hat_vec)
            while cmp_u128_u16impl(tmp, r) == 1:
                q_hat -= 1
                q_hat_hi = q_hat >> 16
                q_hat_lo = q_hat & 0xffff
                q_hat_vec[0] = ti.u16(q_hat_lo)
                q_hat_vec[1] = ti.u16(q_hat_hi)
                count += 1
                tmp = mul_u128_u16impl_82(b_norm, q_hat_vec)
            
            q = extend_by_digit(q, q_hat_vec)
            
            r = sub_u128_u16impl(r, tmp)
            if shift-shift_a-2-i >= 0:
                r_new = ti.Vector([0]*10, ti.u16)
                r_new[0] = dig_rem[shift-shift_a-2-i]
                
                for j in range(9):
                    r_new[j+1] = r[j]
                
                r = r_new
            else:
                break
            
            i += 1
        
        if back_shift:
            r = bit_shift_down_simple_u16impl(r, norm)
        
        q_ret = ti.Vector([q[i] for i in range(8)], ti.u16)
        r_ret = ti.Vector([r[i] for i in range(8)], ti.u16)
        
        q_retu32 = from_u16_to_u32(q_ret)
        r_retu32 = from_u16_to_u32(r_ret)
    else:
        q_retu32 = ti.Vector([0xffffffff]*6, ti.u32)
        r_retu32 = ti.Vector([0xffffffff]*6, ti.u32)

    return q_retu32, r_retu32, zero_devision

