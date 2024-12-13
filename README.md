# Hybrid QAE E2E

This repository contains the implementation of a Hybrid Quantum Autoencoder (QAE) for End-to-End communication. The model integrates classical deep learning techniques with quantum computing to enhance the performance of communication systems.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Train the Model](#1-train-the-model)
  - [2. Retrain the Model](#2-retrain-the-model)
  - [3. Test the Model](#3-test-the-model)
- [Folder Structure](#folder-structure)
- [License](#license)

## Requirements

The following packages are required to run the code:

- `pennylane`
- `tensorflow`

## Installation

### **Option 1** (Basic Installation):

```bash
pip install tensorflow==2.15.0 pennylane
```

### **Option 2** (For Lightning Plugins with `lightning.qubit`, `lightning.gpu`):

```bash
pip install tensorflow==2.15.0
pip install pennylane pennylane-lightning pennylane-lightning-gpu --upgrade
pip install custatevec-cu12
```

### Train the model
```bash
python main.py --train --channel_type rayleigh
```

### Retrain the model
```bash
python main.py --train --retrain --channel_type rayleigh
```

### Test the model
```bash
python main.py --test --channel_type rayleigh
```
```bash
python main.py --test --num_qiubits 7 --channel_type rayleigh
```

### Run the Inference Time Comparisons (Execution Time)
```bash
python test_runtime.py
```