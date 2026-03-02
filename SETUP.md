# Setup Guide — Windows / Anaconda

## Why pip install fails

Both `pip install chefshatgym` (PyPI old V2) and GitHub URL installs require
Git or a compatible build environment. The fix is a **manual download + local install**.

---

## Full Setup (one-time, ~10 minutes)

### Step 1 — Download ChefsHatGYM manually (no Git needed)

1. Open your browser → go to **https://github.com/pablovin/ChefsHatGYM**
2. Click the green **"Code"** button → **"Download ZIP"**
3. Extract the ZIP. You'll get a folder like:  
   `C:\Users\LENOVO\Downloads\ChefsHatGYM-main\ChefsHatGYM-main`  
   *(GitHub zips have a folder-inside-folder — you want the inner one)*

### Step 2 — Open Anaconda Prompt and create an environment

```bash
conda create -n chefshat python=3.10 -y
conda activate chefshat
```

### Step 3 — Install ChefsHatGYM from the downloaded folder

```bash
pip install "C:\Users\LENOVO\Downloads\ChefsHatGYM-main\ChefsHatGYM-main"
```

> Adjust the path to wherever you extracted the zip.  
> The target folder should contain `pyproject.toml` or `setup.py`.

### Step 4 — Install PyTorch (use conda, not pip, to avoid Windows errors)

```bash
conda install pytorch cpuonly -c pytorch -y
```

### Step 5 — Install remaining packages

```bash
pip install numpy matplotlib scipy
conda install jupyter -y
```

### Step 6 — Launch the notebook

```bash
cd "C:\path\to\your\chefshat_rl folder"
jupyter notebook notebook.ipynb
```

---

## Verify installation

Run this in a notebook cell or Anaconda Prompt Python session:

```python
from ChefsHatGym.rooms.room_local import ChefsHatRoomLocal
from ChefsHatGym.agents.agent_random import AgentRandonLocal
import torch
print("ChefsHatGYM: OK")
print(f"PyTorch {torch.__version__}: OK")
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| "not a Python project" | You're pointing pip at the wrong folder level. Navigate into the zip until you see `pyproject.toml` |
| "No module named ChefsHatGym" | Repeat Step 3 with the correct inner folder path |
| torch install fails in notebook cell | Ignore it — torch is installed via conda in Step 4 and will work fine |
