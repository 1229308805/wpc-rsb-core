# wpc-rsb-core

Project code for the Robust Shape Baseline (RSB) model for wind power curve construction from raw SCADA data.

This repository provides the RSB model implementation together with a self-contained synthetic example for quick inspection and reuse.

## Contents

- `wpc_rsb_core.py`: core model implementation
- `example_synthetic.py`: synthetic-data usage example
- `requirements.txt`: runtime dependencies

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python example_synthetic.py
```

## Python usage

```python
import numpy as np
from wpc_rsb_core import RSBModel

wind = np.linspace(0.0, 25.0, 2000)
power = np.clip(((wind - 3.0) / 9.0), 0.0, 1.0) ** 3 * 1500.0

model = RSBModel()
model.train(wind, power)
pred = model.predict(np.linspace(0.0, 25.0, 200))
```

The exported `RSBModel` alias maps to the original `PI2MFramework` implementation retained from the research codebase.
