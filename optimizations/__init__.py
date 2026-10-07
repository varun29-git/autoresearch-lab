"""Stack throughput optimizations, staged for GPU validation.

Everything in this package targets the trial-004 stack (25 x 2304, Peri-LN
branch output norms + learnable value residual) at a fixed layer count. The
modules are standalone: nothing here is wired into terminus/ training until
optimizations/gpu_check.py certifies it on the H100. See README.md for the
physics accounting each module attacks.
"""
