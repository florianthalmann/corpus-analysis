import os, time
import numpy as np
import pandas as pd

class Data:
    def __init__(self, path, columns):
        self.path = path
        self.columns = columns
    
    def read(self):
        if self.path and os.path.isfile(self.path):
            try:
                return pd.read_csv(self.path)
            except pd.errors.EmptyDataError:
                time.sleep(0.01)
                return self.read()
        return pd.DataFrame([], columns=self.columns)
    
    def rows_exist(self, rows):
        df = self.read()
        return all([(df[df.columns[:len(r)]] == r).all(1).any() for r in rows])
    
    def get_rows(self, rows):
        df = self.read()
        return [df.iloc[np.where((df[df.columns[:len(rows[0])]] == r).all(1))].to_numpy()[0]
            for r in rows]
    
    #lazy: calls rows_func only if no rows beginning with ref_rows exist
    def add_rows(self, ref_rows, rows_func):
        if not self.rows_exist(ref_rows):
            rows =  rows_func()
            data = self.read().append(pd.DataFrame(rows, columns=self.columns),
                ignore_index=True)
            if self.path: data.to_csv(self.path, index=False)
            return rows
        else:
            return self.get_rows(ref_rows)
