import os, time
import numpy as np
import pandas as pd
from .util import flatten

class Data:
    def __init__(self, path, columns):
        self.path = path
        self.columns = columns
        self.data = self.read()
    
    def read(self):
        if self.path and os.path.isfile(self.path):
            return pd.read_csv(self.path)
        return pd.DataFrame([], columns=self.columns)
    
    def rows_exist(self, rows):
        return all([(self.data[self.data.columns[:len(r)]] == r).all(1).any() for r in rows])
    
    def get_rows(self, rows):
        rows = [self.data.iloc[np.where(
            (self.data[self.data.columns[:len(r)]] == r).all(1))].to_numpy()
            for r in rows]
        return [r[0] for r in rows if len(r) > 0]
    
    #lazy: calls rows_func only if no rows beginning with ref_rows exist
    def add_rows(self, rows):
        self.data = self.data.append(pd.DataFrame(rows, columns=self.columns),
            ignore_index=True)
        if self.path: self.data.to_csv(self.path, index=False)
