# SNN_HFO_ECoG
A Spiking Neural Network for finding HFOs in prerecorded ECoG

**⚠️ This package has been superseeded by [kburel/snn-hfo-detection](https://github.com/kburel/snn-hfo-detection). Use the new package instead, it can do everything this could and more!**

---

## Old Instructions
For archival purposes, you can still run this old package if you really need to.
The project uses [poetry](https://python-poetry.org/) to manage its dependencies. You can download it via
```bash
pip install --user poetry
```
then clone this repository, `cd` into it and run
```bash
poetry install
```
Place your data in the folder `ECoG_Data/data_python` in the form of `P<number>/Data_pre_Patient_0<number>.mat`, e.g.:
```bash
SNN_HFO_ECoG/ECoG_Data/data_python/P1/Data_pre_Patient_01.mat
```
then run the code via
```bash
poetry run python Run_Test_SNN_ECoG.py
```