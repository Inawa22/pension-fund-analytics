import sys
import os

# Add parent directory to path so pension_fund package is importable  
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Run the pension fund analytics app
exec(open(os.path.join(parent_dir, "pension_fund", "app.py")).read())
