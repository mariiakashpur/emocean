#!/usr/bin/env python
import sys
import pandas as pd

if len(sys.argv) < 4:
    print "Error: Please specify paths to 2 csv files and a new file"
    sys.exit()
dd = sys.argv[1]
occ = sys.argv[2]
feature_csv = sys.argv[3]

occ_read = pd.read_csv(occ)
dd_read = pd.read_csv(dd)


dd_read.merge(occ_read, how='outer').to_csv(feature_csv, index=False)
