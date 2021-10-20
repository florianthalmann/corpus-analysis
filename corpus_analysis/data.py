import os, time
import numpy as np
import pandas as pd

class Data:
    def __init__(self, path, columns):
        self.path = path
        self.columns = columns
    
    def read(self):
        if os.path.isfile(self.path):
            try:
                return pd.read_csv(self.path)
            except pd.errors.EmptyDataError:
                time.sleep(0.01)
                return self.read()
        return pd.DataFrame([], columns=self.columns)
    
    def rows_exist(self, rows):
        df = self.read()
        return all([(df[df.columns[:len(r)]] == r).all(1).any() for r in rows])
    
    #lazy: calls rows_func only if no rows beginning with ref_rows exist
    def add_rows(self, ref_rows, rows_func):
        if not self.rows_exist(ref_rows):
            rows =  pd.DataFrame(rows_func(), columns=self.columns)
            data = self.read().append(rows, ignore_index=True)
            data.to_csv(self.path, index=False)
            return rows
