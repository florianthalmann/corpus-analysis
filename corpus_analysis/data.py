import os
import numpy as np
import pandas as pd

class Data:
    def __init__(self, path, columns):
        self.path = path
        self.columns = columns
    
    def rows_exist(self, rows):
        if os.path.isfile(self.path):
            df = pd.read_csv(self.path)
            return all([(df[df.columns[:len(r)]] == r).all(1).any()
                for r in rows])
    
    #lazy: calls rows_func only if no rows beginning with ref_rows exist
    def add_rows(self, ref_rows, rows_func):
        if not self.rows_exist(ref_rows):
            rows = rows_func()
            print(rows, self.columns)
            rows = pd.DataFrame(rows, columns=self.columns)
            if os.path.isfile(self.path):
                rows = pd.read_csv(self.path).append(rows, ignore_index=True)
            rows.to_csv(self.path, index=False)
    
    def get_rows(self):
        return pd.read_csv(self.path)
