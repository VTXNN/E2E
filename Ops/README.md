## Setup instructions

```
# run cmake and specify install directory
cmake . -DCMAKE_INSTALL_PREFIX=$PWD/release

# compile and install the plugins
make install

# add install directory to python path
export PYTHONPATH="$PWD/release:$PYTHONPATH"

# Optionally: run tests
ctest
```

## Notes
The code requires a python environment with keras and tensorflow installed. 
It is recommended to use anaconda to create a local environment with relevant 
dependencies.
