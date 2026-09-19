Simple ISA instruction-counts diff tool:
if you dumped ISA to two folders A and B, to see the simple diff use:
```sh
./tools/delta.sh A/*.s B/*.s
```

Example output:

```
~/gpuowl$ ./tools/delta.sh tmp4/5M_0_gfx906.s tmp6/5M_0_gfx906.s
tailFused : s_mov_b32 119				      |	tailFused : s_mov_b32 113
tailFused : v_add_f64 443				      |	tailFused : v_add_f64 437
tailFused : v_mul_f64 176				      |	tailFused : v_mul_f64 170
```

## Source checkers

These run as a prerequisite of every build and by `make check`
(`make check-source` runs them alone, `make check-tools` runs their own tests):

```sh
./tools/check_option_inventory.py   # every -use option the kernels or the host read is in the option table
./tools/check_shufl_index.py        # every optimized LDS shuffle path matches the plain one
```

Neither needs a GPU or a build. Both exit 0 when the pass, 1 if finding an issue, and 2 when the source is not in a form
they can read.
