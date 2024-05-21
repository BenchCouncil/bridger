import pandas as pd
import torch

def parse_cond_func(cond_func):
    """
    Input: condition function -string
    Output: corresponding torch function
    """
    if(cond_func == "equal"):
        torch_func = torch.eq
    elif(cond_func == "greater_equal"):
        torch_func = torch.ge
    elif(cond_func == "greater"):
        torch_func = torch.gt
    elif(cond_func == "less_equal"):
        torch_func = torch.le
    elif(cond_func == "less"):
        torch_func = torch.lt
    elif(cond_func == "not_equal"):
        torch_func = torch.ne
    else:
        print("Unknown condition function")
    return torch_func

class TensorTable:
    """
    Relational algebra table in tensor format
    """
    def __init__(self):
        self.column_name = None
        self.tensor = None

    def from_pandas(self, dataframe):    
        """
        Convert pandas dataframe to tensor table
        Input: pandas dataframe
        Output: tensor table
        """
        self.column_name = list(dataframe.columns.values)
        self.tensor = torch.tensor(dataframe.values)

    def rename(self, new_name_list):
        """
        Rename volumn
        """
        self.column_name = new_name_list

    def parse_column_name(self, column_name):
        """
        Input: A list of column names
        Output: Corresponding indices
        """
        return [self.column_name.index(i) for i in column_name]
    
    def select_column(self, column_name):
        """
        Input: A list of column names
        Output: A tensor only contains these columns 
        """
        indices = self.parse_column_name(column_name)
        indices = torch.tensor(indices)
        return torch.index_select(self.tensor, 1, indices)
         
    def projection(self, selected_columns):
        """
        Projection
        """
        self.column_name = selected_columns
        #self.tensor = torch.unique(self.select_column(selected_columns), sorted=True, dim=0)
        self.tensor = self.select_column(selected_columns)

    def selection(self, cond_func, col1, col2=None, threshold=0):
        """
        Selection
        Input: condition func, including equal, greater_equal, greater, less_equal, less, not_equal
        There are mainly two kinds of condition:
            (1) Compare two columns col1 and col2
            (2) Compare one column col1 with threshold
        Output: rows which meet the condition
        """
        func = parse_cond_func(cond_func)
        if(col2==None):
            index = self.column_name.index(col1)
            compare_data = self.tensor[:, index]
            mask = func(compare_data, threshold)
        else:
            index1 = self.column_name.index(col1)
            index2 = self.column_name.index(col2)
            compare_data1 = self.tensor[:, index1]
            compare_data2 = self.tensor[:, index2] 
            mask = func(compare_data1, compare_data2)           
        indices = torch.nonzero(mask, as_tuple=True)[0]
        self.tensor = torch.index_select(self.tensor, 0, indices)
       
    def groupby(self, column_name):
        """
        Group by
        """
        idx = self.column_name.index(column_name)
        keys = self.tensor[:, idx]
        uniques, counts = keys.unique(return_counts = True)
        print(counts)
        #return torch.split_with_sizes(self.tensor, tuple(counts))
        return uniques, counts
       
    def groupby_count(self, column_name):
        """
        Group by and count key
        """
        idx = self.column_name.index(column_name)
        keys = self.tensor[:, idx]
        uniques, counts = keys.unique(return_counts = True)
        out_table = TensorTable()
        out_table.tensor = torch.cat((uniques, counts), -1)
        out_table.column_name = [column_name, "count"]
        return out_table

    def aggregration(self, agg_func, column_name):
        """
        Aggregration
        agg_func : aggregration function, including sum, max, min
        column_name: the name of column to be aggregrated
        """
        idx = self.column_name.index(column_name)
        func_map = {"sum":torch.sum, "max":torch.max, "min":torch.min}
        torch_func = func_map[agg_func] 
        agg_res = torch_func(self.tensor[:, idx], dim=-1)
        return agg_res
 