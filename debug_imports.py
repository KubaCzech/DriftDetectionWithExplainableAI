import sys
import os

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

print("Attempting to import src.datasets...")
try:
    import src.datasets
    print("Import successful!")
    print("Datasets:", src.datasets.DATASETS.keys())
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
