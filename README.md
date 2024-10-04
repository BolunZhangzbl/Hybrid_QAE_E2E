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

- `pennylane==0.36.0`
- `tensorflow==2.8.1`

### Installation

You can install the required packages using pip:

```bash
pip install pennylane==0.36.0 tensorflow==2.8.1
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