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

Implements a 128-bit unsigned integer type used for the mantissa of `f192_t`.
⚠️ **Not intended for direct use**, as:

* It lacks its own operator-swapping decorator.
* It assumes 192-bit float structure and avoids touching the last two limbs (sign and exponent).

### `float192.py`

Implements basic arithmetic operations:

* Addition
* Subtraction
* Multiplication
* Division

Other operations (e.g., comparisons, trigonometric functions) are **not implemented**.

### `ast_transformer.py`

Implements a simple type annotator and BinOpTransformer for swapping binary operators like:

```
a+b -> float192.add_f192(a, b)
```

---

## Performance & Extendability

While it's technically possible to extend this to even higher-precision floats, **Taichi's long compilation times** already make this module impractical for serious performance use. Still, it stands as a **proof of concept** for implementing custom operator overloading in Taichi.

---

## Testing

See `full_test.py` for an example usage and test suite.
This file was used during development and should be sufficient for basic correctness validation, though **minor bugs may still be present**.

---

Let me know if you'd like a badge section or installation instructions added as well.
