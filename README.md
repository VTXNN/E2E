# VTXNN
Neural network for find primary vertices at L1 and their associated tracks

## Setup

1. Install miniconda environment: ```./Env/setupEnv.sh Env/environment.yml Env/env```
2. Activate environment: ```source Env/env.sh```
3. Compile custom tensorflow operations: 
  ```
  mkdir Ops/build
  cd Ops/build
  cmake .. -DCMAKE_INSTALL_PREFIX=../release
  make && make install
  ```
4. Add custom ops to python: ```export PYTHONPATH=<release_dir>:$PYTHONPATH``` (use the absolute path here)
5. Verify that the ops can be imported: ```python -c "import vtxops"``` (should exit without any errors)

For the following steps always make sure that the miniconda environment is active (1.) and that python can find the compiled custom ops (4.).

## Training

1. Convert the track information into TFrecord file format: ```Train/convertTFRecords.py```
2. Train the neural network (ideally on cluster) using the converted files: ```Train/train.py```
3. The main parts of the NN can be exported separately for HLS4ML: ```Train/exportNN.py```

Rinse and repeat.

## Model architecture

Several NN models are defined under: ```Train/vtx/nn```. The default model is ```E2ERef.py```.
