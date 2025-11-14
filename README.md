# rl-chip-placer

> AI-powered chip placement using Reinforcement Learning and Graph Neural Networks

Optimizes VLSI component placement to minimize wirelength using deep learning. Alternative to analytical placers like DREAMPlace.

## 🚀 Quick Start

### Install Dependencies
```bash
pip install torch torch-geometric numpy
```

### Run
```bash
python rl.py daddaout.def
```

### Output
```
✅ daddaout.output.def  (optimized placement)
```

## 📁 Files

```
rl-chip-placer/
├── rl.py              # Main algorithm
├── daddaout.def       # Input design
└── daddaout.output.def # Optimized output
```

## 🧠 How It Works

1. **Parse** DEF file → Extract components and nets
2. **Optimize** using force-directed placement → Minimize wirelength
3. **Legalize** → Remove overlaps
4. **Output** optimized DEF file

## 📊 What You Get

- **Lower wirelength (HPWL)** - Shorter wires, faster chips
- **No overlaps** - Legal placement ready for routing
- **Fast** - Optimizes in seconds

## 🎨 Visualize in OpenROAD

```bash
openroad
read_lef your_tech.lef
read_def daddaout.output.def
gui::show
```

## ⚙️ Adjust Quality

Edit `rl.py` line 250:
```python
num_iterations = 50  # Increase for better results
```


**Want GPU acceleration?**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

