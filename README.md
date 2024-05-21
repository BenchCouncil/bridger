# bridger
A unified framework to bridge the gap between domain-specific frameworks and multiple hardware devices, supporting deep learning, classical machine learning, and data analysis across X86, ARM, RISC-V, IoT devices, and GPU.


## Installation
1. Install tvm, following https://tvm.apache.org/docs/install/index.html

2. `
git clone git@github.com:warmth1905/bridger.git
`

3. Add

`
export UCAB_HOME=/path/of/ucab
`

`
export PYTHONPATH=$UCAB_HOME/python:${PYTHONPATH}
`

`
export TENSORRA_HOME=/path/of/tensor_ra
`

`
export PYTHONPATH=$TENSORRA_HOME/python:${PYTHONPATH}
`

to bashrc
