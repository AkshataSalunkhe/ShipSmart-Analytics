import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _nb_builder import build_notebook

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(os.path.dirname(SCRIPTS_DIR), "notebooks")
os.makedirs(NB_DIR, exist_ok=True)

MAPPING = [
    ("gen_01_data.py", "01_data_generation.ipynb"),
    ("gen_02_segmentation.py", "02_customer_segmentation.ipynb"),
    ("gen_03_wtp.py", "03_wtp_elasticity_model.ipynb"),
    ("gen_04_churn.py", "04_churn_prediction.ipynb"),
    ("gen_05_ab_testing.py", "05_ab_testing_experiment.ipynb"),
    ("gen_06_forecasting.py", "06_revenue_forecasting.ipynb"),
]

if __name__ == "__main__":
    os.chdir(SCRIPTS_DIR)
    for py_name, ipynb_name in MAPPING:
        py_path = os.path.join(SCRIPTS_DIR, py_name)
        ipynb_path = os.path.join(NB_DIR, ipynb_name)
        print(f"\n=== Building {ipynb_name} from {py_name} ===")
        build_notebook(py_path, ipynb_path)
