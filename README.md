```bash
git clone git@github.com:sujingbo0217/ADS_metrics_eval.git
conda create -n vnv python=3.12 -y

pip install -r requirements.txt
conda install -c nvidia/label/cuda-12.6.3 cuda-toolkit -y

cd ADS_metrics_eval
mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=$(which nvcc) \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -Dpybind11_DIR=$(python -m pybind11 --cmakedir)

make -j$(nproc)

cd ADS_metrics_eval
bash run_exp.sh
```

#### Rough experimental results

```
============================================================
Space: inception_v3
============================================================
  real: N=7518  D=2048  dist=DistanceType.L2

  k = 10
    real-only k-NN (N=7518, k=10) ... 0.3s
    level=kitti_sim_images  N_q=21260 ... 2.3s  Bary=12.3303  LID=11.89  Ovlp=0.9875  NMI=0.1881  ARI=0.0069

  k = 20
    real-only k-NN (N=7518, k=20) ... 0.0s
    level=kitti_sim_images  N_q=21260 ... 3.7s  Bary=12.0751  LID=9.59  Ovlp=0.9724  NMI=0.2057  ARI=0.0128

  k = 50
    real-only k-NN (N=7518, k=50) ... 0.0s
    level=kitti_sim_images  N_q=21260 ... 8.4s  Bary=12.1054  LID=7.95  Ovlp=0.9412  NMI=0.2227  ARI=0.0264

============================================================
Space: clip_vit_b32
============================================================
  real: N=7518  D=512  dist=DistanceType.InnerProduct

  k = 10
    real-only k-NN (N=7518, k=10) ... 0.0s
    level=kitti_sim_images  N_q=21260 ... 2.4s  Bary=0.5339  LID=0.00  Ovlp=0.9999  NMI=0.1900  ARI=0.0068

  k = 20
    real-only k-NN (N=7518, k=20) ... 0.0s
    level=kitti_sim_images  N_q=21260 ... 3.2s  Bary=0.5323  LID=0.00  Ovlp=0.9998  NMI=0.2078  ARI=0.0111

  k = 50
    real-only k-NN (N=7518, k=50) ... 0.0s
    level=kitti_sim_images  N_q=21260 ... 6.6s  Bary=0.5338  LID=0.00  Ovlp=0.9994  NMI=0.2399  ARI=0.0232

============================================================
Space: resnet50
============================================================
  real: N=7518  D=2048  dist=DistanceType.L2

  k = 10
    real-only k-NN (N=7518, k=10) ... 0.0s
    level=kitti_sim_images  N_q=21260 ... 2.7s  Bary=10.0681  LID=12.42  Ovlp=0.9984  NMI=0.1887  ARI=0.0063

  k = 20
    real-only k-NN (N=7518, k=20) ... 0.0s
    level=kitti_sim_images  N_q=21260 ... 3.7s  Bary=9.9872  LID=9.72  Ovlp=0.9954  NMI=0.2066  ARI=0.0104

  k = 50
    real-only k-NN (N=7518, k=50) ... 0.0s
    level=kitti_sim_images  N_q=21260 ... 7.3s  Bary=9.9240  LID=7.70  Ovlp=0.9904  NMI=0.2396  ARI=0.0229

============================================================
Space: lpips_vgg
============================================================
  real: N=7518  D=1024  dist=DistanceType.L2

  k = 10
    real-only k-NN (N=7518, k=10) ... 0.0s
    level=kitti_sim_images  N_q=21260 ... 2.6s  Bary=6.5999  LID=15.35  Ovlp=0.9950  NMI=0.1874  ARI=0.0060

  k = 20
    real-only k-NN (N=7518, k=20) ... 0.0s
    level=kitti_sim_images  N_q=21260 ... 4.1s  Bary=6.4159  LID=11.41  Ovlp=0.9740  NMI=0.1988  ARI=0.0081

  k = 50
    real-only k-NN (N=7518, k=50) ... 0.0s
    level=kitti_sim_images  N_q=21260 ... 6.1s  Bary=6.3091  LID=8.43  Ovlp=0.9098  NMI=0.2206  ARI=0.0151

============================================================
Space: pixel
============================================================
  real: N=7518  D=3072  dist=DistanceType.L2

  k = 10
    real-only k-NN (N=7518, k=10) ... 0.0s
    level=kitti_sim_images  N_q=21260 ... 3.1s  Bary=11.6942  LID=9.22  Ovlp=0.8998  NMI=0.1854  ARI=0.0063

  k = 20
    real-only k-NN (N=7518, k=20) ... 0.0s
    level=kitti_sim_images  N_q=21260 ... 3.8s  Bary=11.4475  LID=6.98  Ovlp=0.8276  NMI=0.1872  ARI=0.0089

  k = 50
    real-only k-NN (N=7518, k=50) ... 0.0s
    level=kitti_sim_images  N_q=21260 ... 5.9s  Bary=11.2654  LID=5.30  Ovlp=0.7237  NMI=0.1977  ARI=0.0158

============================================================
Space: segformer
============================================================
  real: N=7518  D=256  dist=DistanceType.L2

  k = 10
    real-only k-NN (N=7518, k=10) ... 0.0s
    level=kitti_sim_images  N_q=21260 ... 2.3s  Bary=10.4630  LID=14.36  Ovlp=0.9999  NMI=0.1972  ARI=0.0081

  k = 20
    real-only k-NN (N=7518, k=20) ... 0.0s
    level=kitti_sim_images  N_q=21260 ... 3.8s  Bary=10.3914  LID=11.02  Ovlp=0.9999  NMI=0.2200  ARI=0.0148

  k = 50
    real-only k-NN (N=7518, k=50) ... 0.0s
    level=kitti_sim_images  N_q=21260 ... 7.8s  Bary=10.4664  LID=8.67  Ovlp=0.9994  NMI=0.2562  ARI=0.0324

============================================================
Post-sweep analyses
============================================================

────────────────────────────────────────────────────────────────────────────────
SUMMARY
────────────────────────────────────────────────────────────────────────────────
Space            k Level              Bary       LID    Ovlp     NMI      ARI
────────────────────────────────────────────────────────────────────────────────
clip_vit_b32    10 kitti_sim_images   0.5339     0.00   0.9999   0.1900   0.0068
clip_vit_b32    20 kitti_sim_images   0.5323     0.00   0.9998   0.2078   0.0111
clip_vit_b32    50 kitti_sim_images   0.5338     0.00   0.9994   0.2399   0.0232
inception_v3    10 kitti_sim_images  12.3303    11.89   0.9875   0.1881   0.0069
inception_v3    20 kitti_sim_images  12.0751     9.59   0.9724   0.2057   0.0128
inception_v3    50 kitti_sim_images  12.1054     7.95   0.9412   0.2227   0.0264
lpips_vgg       10 kitti_sim_images   6.5999    15.35   0.9950   0.1874   0.0060
lpips_vgg       20 kitti_sim_images   6.4159    11.41   0.9740   0.1988   0.0081
lpips_vgg       50 kitti_sim_images   6.3091     8.43   0.9098   0.2206   0.0151
pixel           10 kitti_sim_images  11.6942     9.22   0.8998   0.1854   0.0063
pixel           20 kitti_sim_images  11.4475     6.98   0.8276   0.1872   0.0089
pixel           50 kitti_sim_images  11.2654     5.30   0.7237   0.1977   0.0158
resnet50        10 kitti_sim_images  10.0681    12.42   0.9984   0.1887   0.0063
resnet50        20 kitti_sim_images   9.9872     9.72   0.9954   0.2066   0.0104
resnet50        50 kitti_sim_images   9.9240     7.70   0.9904   0.2396   0.0229
segformer       10 kitti_sim_images  10.4630    14.36   0.9999   0.1972   0.0081
segformer       20 kitti_sim_images  10.3914    11.02   0.9999   0.2200   0.0148
segformer       50 kitti_sim_images  10.4664     8.67   0.9994   0.2562   0.0324
```

#### Mixed test results

```
============================================================
Space: inception_v3
============================================================
  real: N=7518  D=2048  dist=DistanceType.L2

  k = 10
    real-only k-NN (N=7518, k=10) ... 0.306s
    level=kitti_pool_images  N_q=28778 ... 3.8s  Bary=11.0616  LID=0.26  Ovlp=0.4553  NMI=0.0691  ARI=0.0027
    level=kitti_sim_images  N_q=21260 ... 2.4s  Bary=12.3303  LID=11.89  Ovlp=0.9875  NMI=0.1881  ARI=0.0069

  k = 20
    real-only k-NN (N=7518, k=20) ... 0.020s
    level=kitti_pool_images  N_q=28778 ... 5.0s  Bary=11.0352  LID=0.29  Ovlp=0.4719  NMI=0.0755  ARI=0.0043
    level=kitti_sim_images  N_q=21260 ... 4.0s  Bary=12.0751  LID=9.59  Ovlp=0.9724  NMI=0.2057  ARI=0.0128

  k = 50
    real-only k-NN (N=7518, k=50) ... 0.023s
    level=kitti_pool_images  N_q=28778 ... 13.4s  Bary=11.2977  LID=0.45  Ovlp=0.4785  NMI=0.0844  ARI=0.0058
    level=kitti_sim_images  N_q=21260 ... 8.6s  Bary=12.1054  LID=7.95  Ovlp=0.9412  NMI=0.2227  ARI=0.0264

============================================================
Space: clip_vit_b32
============================================================
  real: N=7518  D=512  dist=DistanceType.InnerProduct

  k = 10
    real-only k-NN (N=7518, k=10) ... 0.010s
    level=kitti_pool_images  N_q=28778 ... 3.3s  Bary=0.4339  LID=0.00  Ovlp=0.4572  NMI=0.0694  ARI=0.0024
    level=kitti_sim_images  N_q=21260 ... 2.3s  Bary=0.5339  LID=0.00  Ovlp=0.9999  NMI=0.1900  ARI=0.0068

  k = 20
    real-only k-NN (N=7518, k=20) ... 0.010s
    level=kitti_pool_images  N_q=28778 ... 4.6s  Bary=0.4392  LID=0.00  Ovlp=0.4770  NMI=0.0785  ARI=0.0041
    level=kitti_sim_images  N_q=21260 ... 3.3s  Bary=0.5323  LID=0.00  Ovlp=0.9998  NMI=0.2078  ARI=0.0111

  k = 50
    real-only k-NN (N=7518, k=50) ... 0.012s
    level=kitti_pool_images  N_q=28778 ... 10.9s  Bary=0.4487  LID=0.00  Ovlp=0.4902  NMI=0.0891  ARI=0.0061
    level=kitti_sim_images  N_q=21260 ... 7.2s  Bary=0.5338  LID=0.00  Ovlp=0.9994  NMI=0.2399  ARI=0.0232

============================================================
Space: resnet50
============================================================
  real: N=7518  D=2048  dist=DistanceType.L2

  k = 10
    real-only k-NN (N=7518, k=10) ... 0.020s
    level=kitti_pool_images  N_q=28778 ... 3.9s  Bary=8.6135  LID=0.30  Ovlp=0.4565  NMI=0.0702  ARI=0.0024
    level=kitti_sim_images  N_q=21260 ... 2.6s  Bary=10.0681  LID=12.42  Ovlp=0.9984  NMI=0.1887  ARI=0.0063

  k = 20
    real-only k-NN (N=7518, k=20) ... 0.020s
    level=kitti_pool_images  N_q=28778 ... 5.2s  Bary=8.7000  LID=0.33  Ovlp=0.4759  NMI=0.0766  ARI=0.0033
    level=kitti_sim_images  N_q=21260 ... 3.7s  Bary=9.9872  LID=9.72  Ovlp=0.9954  NMI=0.2066  ARI=0.0104

  k = 50
    real-only k-NN (N=7518, k=50) ... 0.022s
    level=kitti_pool_images  N_q=28778 ... 10.1s  Bary=8.8294  LID=0.47  Ovlp=0.4879  NMI=0.0884  ARI=0.0062
    level=kitti_sim_images  N_q=21260 ... 7.5s  Bary=9.9240  LID=7.70  Ovlp=0.9904  NMI=0.2396  ARI=0.0229

============================================================
Space: lpips_vgg
============================================================
  real: N=7518  D=1024  dist=DistanceType.L2

  k = 10
    real-only k-NN (N=7518, k=10) ... 0.014s
    level=kitti_pool_images  N_q=28778 ... 3.7s  Bary=7.4236  LID=14.46  Ovlp=0.9955  NMI=0.1614  ARI=0.0037
    level=kitti_sim_images  N_q=21260 ... 2.6s  Bary=6.5999  LID=15.35  Ovlp=0.9950  NMI=0.1874  ARI=0.0060

  k = 20
    real-only k-NN (N=7518, k=20) ... 0.014s
    level=kitti_pool_images  N_q=28778 ... 5.2s  Bary=7.2855  LID=10.18  Ovlp=0.9793  NMI=0.1730  ARI=0.0054
    level=kitti_sim_images  N_q=21260 ... 4.2s  Bary=6.4159  LID=11.41  Ovlp=0.9740  NMI=0.1988  ARI=0.0081

  k = 50
    real-only k-NN (N=7518, k=50) ... 0.017s
    level=kitti_pool_images  N_q=28778 ... 9.6s  Bary=7.1765  LID=7.04  Ovlp=0.9356  NMI=0.1973  ARI=0.0135
    level=kitti_sim_images  N_q=21260 ... 6.4s  Bary=6.3091  LID=8.43  Ovlp=0.9098  NMI=0.2206  ARI=0.0151

============================================================
Space: pixel
============================================================
  real: N=7518  D=3072  dist=DistanceType.L2

  k = 10
    real-only k-NN (N=7518, k=10) ... 0.028s
    level=kitti_pool_images  N_q=28778 ... 4.1s  Bary=10.7074  LID=0.30  Ovlp=0.4419  NMI=0.0700  ARI=0.0020
    level=kitti_sim_images  N_q=21260 ... 3.0s  Bary=11.6942  LID=9.22  Ovlp=0.8998  NMI=0.1854  ARI=0.0063

  k = 20
    real-only k-NN (N=7518, k=20) ... 0.028s
    level=kitti_pool_images  N_q=28778 ... 5.4s  Bary=10.7853  LID=0.32  Ovlp=0.4345  NMI=0.0739  ARI=0.0024
    level=kitti_sim_images  N_q=21260 ... 3.9s  Bary=11.4475  LID=6.98  Ovlp=0.8276  NMI=0.1872  ARI=0.0089

  k = 50
    real-only k-NN (N=7518, k=50) ... 0.029s
    level=kitti_pool_images  N_q=28778 ... 8.2s  Bary=10.8987  LID=0.46  Ovlp=0.4036  NMI=0.0783  ARI=0.0020
    level=kitti_sim_images  N_q=21260 ... 6.3s  Bary=11.2654  LID=5.30  Ovlp=0.7237  NMI=0.1977  ARI=0.0158

============================================================
Space: segformer
============================================================
  real: N=7518  D=256  dist=DistanceType.L2

  k = 10
    real-only k-NN (N=7518, k=10) ... 0.009s
    level=kitti_pool_images  N_q=28778 ... 3.3s  Bary=8.7842  LID=0.29  Ovlp=0.4576  NMI=0.0719  ARI=0.0029
    level=kitti_sim_images  N_q=21260 ... 2.3s  Bary=10.4630  LID=14.36  Ovlp=0.9999  NMI=0.1972  ARI=0.0081

  k = 20
    real-only k-NN (N=7518, k=20) ... 0.009s
    level=kitti_pool_images  N_q=28778 ... 5.8s  Bary=8.8892  LID=0.31  Ovlp=0.4773  NMI=0.0805  ARI=0.0052
    level=kitti_sim_images  N_q=21260 ... 4.0s  Bary=10.3914  LID=11.02  Ovlp=0.9999  NMI=0.2200  ARI=0.0148

  k = 50
    real-only k-NN (N=7518, k=50) ... 0.012s
    level=kitti_pool_images  N_q=28778 ... 13.3s  Bary=9.1536  LID=0.45  Ovlp=0.4903  NMI=0.0943  ARI=0.0099
    level=kitti_sim_images  N_q=21260 ... 8.2s  Bary=10.4664  LID=8.67  Ovlp=0.9994  NMI=0.2562  ARI=0.0324

============================================================
Post-sweep analyses
============================================================

────────────────────────────────────────────────────────────────────────────────
SUMMARY
────────────────────────────────────────────────────────────────────────────────
Space            k Level               Bary      LID     Ovlp     NMI     ARI
────────────────────────────────────────────────────────────────────────────────
clip_vit_b32    10 kitti_pool_images   0.4339     0.00   0.4572   0.0694   0.0024
clip_vit_b32    10 kitti_sim_images   0.5339     0.00   0.9999   0.1900   0.0068
clip_vit_b32    20 kitti_pool_images   0.4392     0.00   0.4770   0.0785   0.0041
clip_vit_b32    20 kitti_sim_images   0.5323     0.00   0.9998   0.2078   0.0111
clip_vit_b32    50 kitti_pool_images   0.4487     0.00   0.4902   0.0891   0.0061
clip_vit_b32    50 kitti_sim_images   0.5338     0.00   0.9994   0.2399   0.0232
inception_v3    10 kitti_pool_images  11.0616     0.26   0.4553   0.0691   0.0027
inception_v3    10 kitti_sim_images  12.3303    11.89   0.9875   0.1881   0.0069
inception_v3    20 kitti_pool_images  11.0352     0.29   0.4719   0.0755   0.0043
inception_v3    20 kitti_sim_images  12.0751     9.59   0.9724   0.2057   0.0128
inception_v3    50 kitti_pool_images  11.2977     0.45   0.4785   0.0844   0.0058
inception_v3    50 kitti_sim_images  12.1054     7.95   0.9412   0.2227   0.0264
lpips_vgg       10 kitti_pool_images   7.4236    14.46   0.9955   0.1614   0.0037
lpips_vgg       10 kitti_sim_images   6.5999    15.35   0.9950   0.1874   0.0060
lpips_vgg       20 kitti_pool_images   7.2855    10.18   0.9793   0.1730   0.0054
lpips_vgg       20 kitti_sim_images   6.4159    11.41   0.9740   0.1988   0.0081
lpips_vgg       50 kitti_pool_images   7.1765     7.04   0.9356   0.1973   0.0135
lpips_vgg       50 kitti_sim_images   6.3091     8.43   0.9098   0.2206   0.0151
pixel           10 kitti_pool_images  10.7074     0.30   0.4419   0.0700   0.0020
pixel           10 kitti_sim_images  11.6942     9.22   0.8998   0.1854   0.0063
pixel           20 kitti_pool_images  10.7853     0.32   0.4345   0.0739   0.0024
pixel           20 kitti_sim_images  11.4475     6.98   0.8276   0.1872   0.0089
pixel           50 kitti_pool_images  10.8987     0.46   0.4036   0.0783   0.0020
pixel           50 kitti_sim_images  11.2654     5.30   0.7237   0.1977   0.0158
resnet50        10 kitti_pool_images   8.6135     0.30   0.4565   0.0702   0.0024
resnet50        10 kitti_sim_images  10.0681    12.42   0.9984   0.1887   0.0063
resnet50        20 kitti_pool_images   8.7000     0.33   0.4759   0.0766   0.0033
resnet50        20 kitti_sim_images   9.9872     9.72   0.9954   0.2066   0.0104
resnet50        50 kitti_pool_images   8.8294     0.47   0.4879   0.0884   0.0062
resnet50        50 kitti_sim_images   9.9240     7.70   0.9904   0.2396   0.0229
segformer       10 kitti_pool_images   8.7842     0.29   0.4576   0.0719   0.0029
segformer       10 kitti_sim_images  10.4630    14.36   0.9999   0.1972   0.0081
segformer       20 kitti_pool_images   8.8892     0.31   0.4773   0.0805   0.0052
segformer       20 kitti_sim_images  10.3914    11.02   0.9999   0.2200   0.0148
segformer       50 kitti_pool_images   9.1536     0.45   0.4903   0.0943   0.0099
segformer       50 kitti_sim_images  10.4664     8.67   0.9994   0.2562   0.0324
```
