# Scalable Parallel Feature Extraction and Tracking for Large Time-Varying 3D Volume Data
Our implementation of the paper Scalable Parallel Feature Extraction and Tracking for Large Time-varying 3D Volume Data. We built this as a part of our course CS677: Topics in Large Data Analysis and Visualization

To run the code the following libraries need to be present:
  - A CPP compiler
  - MPI library installation: We have used the mpich library to run our code

The script run_script.sh automates the running on the kd lab systems where the hosts can be configured.

## Installation and Running

Make a directory named build inside the project folder

```bash
mkdir build
cd build
```

We have used cmake to build the Makefile which needs to be installed. Inside the build directory, run
```bash 
cmake ..
```

After this, the Makefile will be formed which can be run directly using make.
```bash 
make
```

Once the binary is built, we can directly run the run_script.sh in the parent directory. Be sure to make the PATH variables point correctly.

Now run the following command

```
../run_script.sh
```

## python file information

- `binary_merger.py` merges all the binary outputs of each of the processors
- `binary_to_vti.py` for converting binary files to .vti file
- `vti_to_binary.py` for converting .vti to binary
- ``

