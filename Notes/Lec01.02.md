# Pytorch Semantics

## Tensors Memory

- float32 (fp32/single precision): 1 sign + 8 exponent + 23 fraction

```python
x = torch.zeros(4, 8)  # Matrix 4*8
assert x.dtype = torch.float32  # Default type
assert x.numel() == 4 * 8
assert x.element_size() == 4  # Float is 4 bytes
assert get_memory_usage(x) == 4 * 8 * 4  # 128 bytes
```

- Double precision fp64/float64
- float16 (fp16): 1 sign + 5 exponent + 10 fraction 
- bfloat16 (bf16): 1 sign + 8 exponent + 7 fraction 
- fp8 (E4M3): 1 sign + 4 exponent + 3 fraction 
- fp8 (E5M2): 1 sign + 5 exponent + 2 fraction
- fp4: 1 sign + 2 exponent + 1 fraction (15 candidates)


**Mixed precision training:**
- Use bf16 for parameters, activations, and gradients
- Use fp32 for optimizer states


## Einops

Einops是一个看起来很不错的语法糖，这里列一些课堂上提到的用法：

```python
x = torch.ones(3, 4)  # seq1 hidden 
y = torch.ones(4, 3)  # hidden seq2 
# Old way
z = x @ y   # seq1 seq2  
# New (einops) way
z = einsum(x, y, "seq1 hidden, hidden seq2 -> seq1 seq2")  # 给每一列定一个名字 然后可以算乘法，将重名作为内联，直接抵消
```
**列名可以自动换位，不用transpose**

```python
x = torch.ones(2, 3, 4)  # batch seq1 hidden 
y = torch.ones(2, 3, 4)  # batch seq2 hidden 
# Old way
z = x @ y.transpose(-2, -1)  # batch seq1 seq2  
# New (einops) way
z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")  
Dimensions that are not named in the output are summed over.
# Or can use `...` to represent broadcasting over any number of dimensions
z = einsum(x, y, "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2")  # ...可以抵消不止一列
```

**rebuce这种对单独列的操作也比较不错**

```python
x = torch.ones(2, 3, 4)  # batch seq hidden 
# Old way
y = x.sum(dim=-1)  
# New (einops) way
y = reduce(x, "... hidden -> ...", "sum")
```