"""gfx942 producers emitting BF16 outputs and exact rowwise FNUZ sidecars."""
import torch
import triton
import triton.language as tl

from .fp8_kernels import FP8, FP8_MAX, _scale


@triton.jit
def _emit(Y, C, S, ROW, COL, WIDTH:tl.constexpr, ROWS:tl.constexpr, BLOCK:tl.constexpr):
    blocks=tl.reshape(Y.to(tl.float32),(BLOCK//128,128))
    scales=_scale(tl.max(tl.abs(blocks),1),1e-5,240.)
    codes=tl.div_rn(blocks,scales[:,None])
    codes=tl.where((tl.abs(blocks)<float('inf')) & (tl.abs(codes)<float('inf')),
                  tl.minimum(tl.maximum(codes,-240.),240.),float('nan'))
    tl.store(C+ROW*WIDTH+COL,tl.reshape(codes,(BLOCK,)).to(C.dtype.element_ty),COL<WIDTH)
    groups=tl.arange(0,BLOCK//128)
    tl.store(S+groups*ROWS+ROW,scales,groups<WIDTH//128)


@triton.jit
def _norm(X,W,Y,R,C,S,ROWS:tl.constexpr,WIDTH:tl.constexpr,GAIN:tl.constexpr,EPS:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0).to(tl.int64);col=tl.arange(0,BLOCK)
    x=tl.load(X+row*WIDTH+col,col<WIDTH,other=0).to(tl.float32)
    inverse=tl.rsqrt(tl.sum(x*x,0)/WIDTH+EPS)
    y=x*inverse
    if GAIN:
        weight=tl.load(W+(col if GAIN==2 else tl.full((BLOCK,),0,tl.int32)),col<WIDTH,other=0).to(tl.float32)
        y*=weight
    rounded=y.to(Y.dtype.element_ty)
    tl.store(Y+row*WIDTH+col,rounded,col<WIDTH);tl.store(R+row,inverse)
    _emit(rounded,C,S,row,col,WIDTH,ROWS,BLOCK)


@triton.jit
def _swiglu(X,W,Y,R,C,S,ROWS:tl.constexpr,WIDTH:tl.constexpr,NORM:tl.constexpr,EPS:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0).to(tl.int64);col=tl.arange(0,BLOCK)
    g=tl.load(X+row*(2*WIDTH)+col,col<WIDTH,other=0).to(tl.float32)
    u=tl.load(X+row*(2*WIDTH)+WIDTH+col,col<WIDTH,other=0).to(tl.float32)
    sigmoid=1./(1.+tl.exp(-g))
    silu=(g*sigmoid).to(X.dtype.element_ty).to(tl.float32)
    product=(silu*u).to(X.dtype.element_ty).to(tl.float32)
    if NORM:
        inverse=tl.rsqrt(tl.sum(product*product,0)/WIDTH+EPS)
        product=(product*inverse)*tl.load(W);tl.store(R+row,inverse)
    rounded=product.to(Y.dtype.element_ty)
    tl.store(Y+row*WIDTH+col,rounded,col<WIDTH)
    _emit(rounded,C,S,row,col,WIDTH,ROWS,BLOCK)


def _outputs(rows,width,device):
    if width%128 or rows<=0:raise ValueError('FP8 producers require nonempty aligned matrices')
    y=torch.empty((rows,width),device=device,dtype=torch.bfloat16)
    codes=torch.empty((rows,width),device=device,dtype=FP8)
    scales=torch.empty((width//128,rows),device=device,dtype=torch.float32).T
    return y,codes,scales


def norm_forward(values,weight,eps):
    flat=values.contiguous().view(-1,values.shape[-1]);rows,width=flat.shape
    y,c,s=_outputs(rows,width,flat.device)
    inverse=torch.empty(rows,device=flat.device,dtype=torch.float32)
    gain=0 if weight is None else (1 if weight.ndim==0 else 2)
    _norm[(rows,)](flat,weight,y,inverse,c,s,rows,width,gain,eps,triton.next_power_of_2(width),enable_fp_fusion=False)
    return y,inverse,flat,(c,s)


def packed_forward(packed,weight,eps):
    width=packed.shape[-1]//2;rows=packed.numel()//(2*width)
    y,c,s=_outputs(rows,width,packed.device)
    inverse=torch.empty(rows,device=packed.device,dtype=torch.float32) if weight is not None else None
    _swiglu[(rows,)](packed,weight,y,inverse,c,s,rows,width,weight is not None,eps,triton.next_power_of_2(width),
                    num_warps=8 if width>4096 else 4,enable_fp_fusion=False)
    return y.reshape(*packed.shape[:-1],width),inverse,(c,s)
