import os, time
import numpy as np
import pandas as pd
from .util import flatten

class Data:
    def __init__(self, path, columns):
        self.path = path
        self.columns = columns
    
    def read(self):
        if self.path and os.path.isfile(self.path):
            try:
                df = pd.read_csv(self.path)
                return (df if len(df) > 0 else self.reread())#for concurrency
            except pd.errors.EmptyDataError:#for concurrency
                return self.reread()
        return pd.DataFrame([], columns=self.columns)
    
    def reread(self):
        time.sleep(0.01)
        return self.read()
    
    def rows_exist(self, rows):
        df = self.read()
        return all([(df[df.columns[:len(r)]] == r).all(1).any() for r in rows])
    
    def get_rows(self, rows):
        df = self.read()
        rows = [df.iloc[np.where((df[df.columns[:len(r)]] == r).all(1))].to_numpy()
            for r in rows]
        return [r[0] for r in rows if len(r) > 0]
    
    #lazy: calls rows_func only if no rows beginning with ref_rows exist
    def add_rows(self, ref_rows, rows_func):
        if not self.rows_exist(ref_rows):
            rows = rows_func()
            data = self.read().append(pd.DataFrame(rows, columns=self.columns),
                ignore_index=True)
            if self.path: data.to_csv(self.path, index=False)
            return rows
        else:
            return self.get_rows(ref_rows)
