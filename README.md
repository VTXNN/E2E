# E2E
End-to-end neural network training

To setup the python environment for Quantised training install anaconda: https://www.anaconda.com/ for your chosen platform then in the Env subdirectory:

`conda env create -f environment.yml`

`conda activate qtf`

Then in the Ops directory:

`cmake . -DCMAKE_INSTALL_PREFIX=$PWD/release`

`make install`

To run a basic training loop first training a unquantised network followed by a quantised network you will need to change line 16 of setup.yaml to
the directory that contains a  Train, Test, Val, MET set of folders each with .tfrecord files for training on
Then you will need a comet_ml account https://www.comet.ml/site/ for logging all the training info and resulting plots and put your API key in line 11
of runTraining.sh 

Also update line 7 of runTraining.sh to the absolute path of the directory Ops/release

Once done you can simply run runTraining.sh which should run the training and evaluation of the model
