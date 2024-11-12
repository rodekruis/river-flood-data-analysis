
import os 
from pathlib import Path
import sys 
cur = Path(os.getcwd())
parent_dir = cur.parent
working_dir = parent_dir #/ 'MaliGloFAS/river_flood_data_analysis'
os.chdir(working_dir)
sys.path.append(working_dir)
print (os.getcwd())