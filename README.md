---

# taichi-float192

This module provides functions to convert between `f192` and `f32`, along with a decorator that enables Taichi functions and kernels to use the `f192_t` type.

---

## The `f192_t` Type

The `f192_t` type is defined as:

```python
ti.types.vector(6, ti.u32)
```

It consists of:

* **Mantissa (128 bits)**: Stored in the first 4 `u32` elements.
* **Sign and optional flags (32 bits)**: Stored in the 5th `u32` element. This could be used, for example, to flag a zero-division error.
* **Exponent (32 bits)**: Stored in the 6th `u32` element, with a bias of `0x80000000`.

---

## Architecture and Usage

The main design goal was to make `f192_t` accessible within the Taichi environment using only 32-bit types. This avoids relying on `uint64`, which can cause backend issues (especially under Vulkan). Vulkan support was the main goal for also having AOT support but Vulkan backend throws obscure errors.

### Compatibility

* `ti.cpu`
* `ti.cuda`
* Vulkan backend will throw unexplained errors, I believe this is a taichi error since syntactically it should work.

### Type Usage

Always use the `f192_t` type for accessing and passing 192-bit floats, even for parameters and return values of `@ti.func`s. This is necessary because:

* Taichi does **not** support operator overloading.
* A **temporary file** is used to **swap operators** with corresponding function calls.
* For this to work reliably, **static typing** is required to identify all `f192_t` usage.

---

## Operator Support via `supports_f192` Decorator

The `supports_f192` decorator handles operator replacement. It:

* Imports the `__main__` module in a temp file.
* Requires you to pass the `globals()` dictionary.
* Therefore, **all usage must be inside an `if __name__ == '__main__':` block**, similar to how Python's `multiprocessing` works.

---

## Supporting Modules

### `mantissa128.py`

Implements a 128-bit unsigned integer type used for the mantissa of `f192_t`. **Not intended for direct use**, as:

* It lacks its own operator-swapping decorator.
* It assumes 192-bit float structure and avoids touching the last two limbs (sign and exponent).

### `float192.py`

Implements basic arithmetic operations:

* Addition
* Subtraction
* Multiplication
* Division

And also implements comparisons, other operations (e.g. trigonometric functions and other mathematical functions) are **not implemented**. It also implements a few convinience features, converting f192 to f32 and creating new f192 from i32 or f32, a pure python implementation of converting string to f192 is also included (this of course cannot be run inside the Taichi scope, since that doesn't support strings).

### `ast_transformer.py`

Implements a simple type annotator and BinOpTransformer for swapping binary operators like:

    a+b -> float192.add_f192(a, b)
This is a key component of this modul since Taichi itself doesnt allow operator overloading, but I do not guarantee that it works in every possible scenario (and any error steming from here will throw obscure errors), however, the basic use cases are covered.

---

## Performance & Extendability

This module can be extended to support multiple even higher precisions (and might be done in the future), with the code being created dynamically at runtime. An initial issue was Taichi's compilation time bloat, but after some digging I fould the solution: ti.real_func, which was added kind of silently to Taichi at 1.7 and lets the functions being compiled separately therefore avoiding the massive compilation time bloat coming from inlining a lot of nested functions inside the kernel. Theoretically a 128 bit mantissa gives a precision of around 38 significant digits with the cost of an almost negligable amount of inconvinience and a theoretically estimated 10-20 times slowdown to that of 32 bit floats (this might be higher, I haven't yet benchmarked it).

---

## Testing

See `full_test.py` for an example usage and test suite, this file was used during development and should be sufficient for basic correctness validation, though **minor bugs may still be present** (I specifically expect the division to be buggy at resolutions where the standard f32 would fail due to zero division, otherwise it should be fine). A nicer test is shown in `mandelbrot test.py` which calculates a small part of the mandelbrot fractal zoomed at such an extent where regular float64 would have visible granularity due to rounding errors in the pixel positions. 

