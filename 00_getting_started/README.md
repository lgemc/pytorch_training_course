## Pytorch

Torch is a library to build and perform the process of convert data into predictive models

## How to start

First you should check if you have cuda available in your system

```bash
nvidia-smi | awk '/CUDA Version/ {for(i=1;i<=NF;i++) if ($i=="CUDA") print $(i+2)}'  
```

If yes you are going to see a response like `12.6`

Next you should install requirements (please edit if required to have the correct cuda version)

```bash
python -m unittest test_binary_operations.py 
```