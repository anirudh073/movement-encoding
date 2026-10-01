import os
import datajoint as dj
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if os.path.basename(os.getcwd()) == 'notebooks':
    os.chdir("..")

dj.config.load("/media/labuser/NA_1_2025/spyglass_common/dj_local_conf.json")

base_dir = "/mnt/loon/spyglass-data/"
raw_dir = base_dir + "raw/"

os.environ["SPYGLASS_BASE_DIR"] = base_dir
os.environ["SPYGLASS_RAW_DIR"] = raw_dir
os.environ["KACHERY_CLOUD_DIR"] = "/home/labuser/.kachery-cloud"
os.environ["KACHERY_ZONE"] = "default"

import spyglass.common as sgc

# Set nwb_file_name, nwb_copy_file_name, session_restrict, and prefix per animal in each notebook.
# Example for wilbur:
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
nwb_file_name = "Mango20211220.nwb"
nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
session_restrict = {"nwb_file_name": nwb_copy_file_name}
prefix = "Mango20211220"
