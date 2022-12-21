# 2IMP40 Empirical Methods in Software Engineering
Download the dataset from [https://zenodo.org/record/7182101]() and put it in the root directory under
`./ThePublicJiraDataset`.

# Setup
Make sure you have `python` and `python-venv` installed.
```bash
python -m venv .venv
# on linux need to check how to activate venv on windows
source .venv/bin/activate
pip install -r requirements.txt
python stats.py
```