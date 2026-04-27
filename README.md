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
