we have memory leaks everywhere

Chaining `reshape()` after `arange()` results in garbage being printed.

```cpp
  auto a = arange<i32>(1, 5).reshape({2, 2});
  print_array(a);
```

However, this works fine

```cpp
  auto a = arange<i32>(1, 5);
  auto b = a.reshape({2, 2});
  print_array(b);
```
