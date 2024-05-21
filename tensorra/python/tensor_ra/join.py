from tracemalloc import start
import torch
from tensor_ra.tensor_table import TensorTable
import time
import torch.multiprocessing as mp
from pandas.core.reshape.merge import get_join_indexers
import numpy as np

def parse_table(table, key):
    """
    Input: table to be joined
    Output: corresponding key tensor
    """
    index = table.column_name.index(key)
    key_tensor = table.tensor[:, index]
    return key_tensor

def sort_key(table1, key1, table2, key2):
    """
    Input: two tables to be joined and corresponding keys
    Output: Sorted keys and corresponding indices
    """
    key_tensor1 = parse_table(table1, key1)
    key_tensor2 = parse_table(table2, key2)
    unique_key1, inverse_idx1, nkey1 = torch.unique(key_tensor1, return_inverse=True, return_counts=True)
    unique_key2, inverse_idx2, nkey2 = torch.unique(key_tensor2, return_inverse=True, return_counts=True)
    return unique_key1, inverse_idx1, nkey1, unique_key2, inverse_idx2, nkey2

class TensorCopy:
    """
    Tensor Copy
    """

    def __init__(self, left_tensor, right_tensor, out_tensor, nprocesses):
        """
        nprocesses: number of processes 
        join_type: join type, incluing "inner_join", "full_join", "left_join", "right_join", "cross_join"
        """
        self.left_tensor = left_tensor
        self.right_tensor = right_tensor
        self.out_tensor = out_tensor
        self.len1 = left_tensor.shape[0]
        self.len2 = right_tensor.shape[0]
        self.col1 = left_tensor.shape[1]
        self.col2 = right_tensor.shape[1]
        if(nprocesses == None):
            self.nprocesses = mp.cpu_count()
        else:
            self.nprocesses = nprocesses

class CrossJoinCopy(TensorCopy):
    """
    Tensor Copy for cross join    
    """

    def __init__(self, left_tensor, right_tensor, out_tensor, nprocesses):
        super(CrossJoinCopy, self).__init__(left_tensor, right_tensor, out_tensor, nprocesses)

    def cross_join_copy(self, start, end):
        for i in range(start, end):
            start_idx = i * self.len2
            end_idx = (i + 1) * self.len2
            self.out_tensor[start_idx:end_idx, :self.col1] = self.left_tensor[i, :]
            self.out_tensor[start_idx:end_idx, self.col1:] = self.right_tensor
    
    def multi_processing_copy(self):
        """
        Cross Join copy used multi processing 
        """
        self.left_tensor.share_memory_()
        self.right_tensor.share_memory_()
        self.out_tensor.share_memory_()
        len_single_process = self.len1 / self.nprocesses
        for rank in range(self.nprocesses):
            start = int(len_single_process * rank)
            end = max(int(len_single_process * (rank + 1)), self.len1)
            p = mp.Process(target=self.cross_join_copy, args=(start, end))
            p.start()

    def single_copy(self):
        """
        Cross Join copy without multi processing
        """
        self.cross_join_copy(0, self.len1)            

class InnerJoinCopy(TensorCopy):
    """
    Tensor Copy for inner join
    """
    def __init__(self, left_tensor, right_tensor, out_tensor, idx_out, inverse_idx1, inverse_idx2, intersection_nkey1, intersection_nkey2, nprocesses): 
        super(InnerJoinCopy, self).__init__(left_tensor, right_tensor, out_tensor, nprocesses)
        self.idx_out = idx_out
        self.inverse_idx1 = inverse_idx1
        self.inverse_idx2 = inverse_idx2
        self.intersection_nkey1 = intersection_nkey1
        self.intersection_nkey2 = intersection_nkey2 
        self.nkey = torch.zeros_like(self.idx_out)

    def left_copy(self, start, end):
        for i in range(start, end):
            idx = self.inverse_idx1[i]
            if(idx >= 0):
                start_idx = self.idx_out[idx] + self.nkey[idx]
                end_idx = start_idx + self.intersection_nkey2[idx]         
                self.out_tensor[start_idx:end_idx, 0:self.col1] = self.left_tensor[i, :]
                self.nkey[idx] = self.nkey[idx] + self.intersection_nkey2[idx]

    def right_copy(self, start, end):
        for i in range(start, end):
            idx = self.inverse_idx2[i]
            if(idx >= 0):
                for j in range(self.intersection_nkey1[idx]):
                    start_idx = self.idx_out[idx] + self.nkey[idx] + j * self.intersection_nkey2[idx]
                    end_idx = start_idx + 1
                    self.out_tensor[start_idx:end_idx, self.col1:] = self.right_tensor[i, :]
                self.nkey[idx] = self.nkey[idx] + 1

    def multi_processing_copy(self):
        """
        Inner Join copy used multi processing 
        """
        self.left_tensor.share_memory_()
        self.right_tensor.share_memory_()
        self.out_tensor.share_memory_()
        self.idx_out.share_memory_()
        self.inverse_idx1.share_memory_()
        self.inverse_idx2.share_memory_()
        self.intersection_nkey1.share_memory_()
        self.intersection_nkey2.share_memory_()
        self.nkey.share_memory_()
        len_single_process = self.len1 / self.nprocesses
        self.left_copy(0, self.len1)
        for rank in range(self.nprocesses):
            start = int(len_single_process * rank)
            end = max(int(len_single_process * (rank + 1)), self.len1)
            p = mp.Process(target=self.left_copy, args=(start, end))
            p.start()
        """
        len_single_process = self.len2 / self.nprocesses
        for rank in range(self.nprocesses):
            start = int(len_single_process * rank)
            end = max(int(len_single_process * (rank + 1)), self.len2)
            p = mp.Process(target=self.right_copy, args=(start, end))
            p.start()
        """
        self.nkey = torch.zeros_like(self.idx_out)
        self.right_copy(0, self.len2)
    def single_copy(self):
        """
        Inner Join copy without mutli processing
        """
        self.left_copy(0, self.len1)
        self.nkey = torch.zeros_like(self.idx_out)
        self.right_copy(0, self.len2)

def inner_join(table1, key1, table2, key2, use_multi_processing=True, nprocesses=None):
    key_tensor1 = parse_table(table1, key1)
    key_tensor2 = parse_table(table2, key2)
    if(key_tensor2.type()!=key_tensor1.type()):
        key_tensor2 = key_tensor2.type_as(key_tensor1)
    key_tensor1 = [key_tensor1.numpy()]
    key_tensor2 = [key_tensor2.numpy()]
    left_key_index, right_key_index = get_join_indexers(key_tensor1, key_tensor2, sort=True, how="inner")
    left_table = torch.index_select(table1.tensor, 0, torch.tensor(left_key_index))
    right_table = torch.index_select(table2.tensor, 0, torch.tensor(right_key_index))
    res_table = TensorTable()
    res_table.column_name = table1.column_name + table2.column_name
    res_table.tensor = torch.cat((left_table, right_table), 1)
    return res_table

"""
def inner_join(table1, key1, table2, key2, use_multi_processing=True, nprocesses=None):
    #Inner join
    a = time.perf_counter()
    unique_key1, inverse_idx1, nkey1, unique_key2, inverse_idx2, nkey2 = sort_key(table1, key1, table2, key2)
    b = time.perf_counter()
    if(unique_key1.type()!=unique_key2.type()):
        unique_key2 = unique_key2.type_as(unique_key1)
    i = 0
    j = 0
    len1 = unique_key1.shape[0]
    len2 = unique_key2.shape[0]
    intersection_key = []
    intersection_nkey1 = []
    intersection_nkey2 = []
    map1 = {}
    map2 = {}
    n_key = 0
    while((i < len1) & (j < len2)):
        if(unique_key1[i] == unique_key2[j]):
            intersection_key.append(unique_key1[i].item())
            intersection_nkey1.append(nkey1[i].item())
            intersection_nkey2.append(nkey2[j].item())
            map1[i] = n_key
            map2[j] = n_key 
            i = i + 1
            j = j + 1 
            n_key = n_key + 1
        elif(unique_key1[i] < unique_key2[j]):
            map1[i] = -1
            i = i + 1
        else:
            map2[j] = -1
            j = j + 1
    intersection_time = time.perf_counter()
    #print(intersection_time - b)
    inverse_idx1.apply_(lambda i: map1.get(i, -1))
    inverse_idx2.apply_(lambda i: map2.get(i, -1))
    intersection_key = torch.tensor(intersection_key)
    intersection_nkey1 = torch.tensor(intersection_nkey1)
    intersection_nkey2 = torch.tensor(intersection_nkey2)
    nkey_out = intersection_nkey1 * intersection_nkey2
    idx_out = torch.cat((torch.tensor([0]), torch.cumsum(nkey_out, dim=0)))
    c = time.perf_counter()
    #print(c - intersection_time)
    res_table = TensorTable()
    res_table.column_name = table1.column_name + table2.column_name
    out_tensor = torch.empty([idx_out[-1], len(res_table.column_name)])
    copy = InnerJoinCopy(table1.tensor, table2.tensor, out_tensor, idx_out, inverse_idx1, inverse_idx2, intersection_nkey1, intersection_nkey2, nprocesses)
    if(use_multi_processing == True):
        copy.multi_processing_copy()
    else:
        copy.single_copy()
    res_table.tensor = out_tensor
    d = time.perf_counter()
    print(b-a, c-b, d-c)
    return res_table
"""
def full_join(table1, key1, table2, key2):
    """
    Full join
    """
    col1 = table1.column_name
    tensor1 = table1.tensor
    col2 = table2.column_name
    tensor2 = table2.tensor
    unique_key1, inverse_idx1, nkey1, unique_key2, inverse_idx2, nkey2 = sort_key(table1, key1, table2, key2)
    if(unique_key1.type()!=unique_key2.type()):
        unique_key2 = unique_key2.type_as(unique_key1)
    #print(unique_key1, unique_key2)
    #print(inverse_idx1, inverse_idx2)
    i = 0
    j = 0
    len1 = unique_key1.shape[0]
    len2 = unique_key2.shape[0]
    union_key = []
    union_nkey1 = []
    union_nkey2 = []
    map1 = {}
    map2 = {}
    n_key = 0
    while((i < len1) & (j < len2)):
        if(unique_key1[i] == unique_key2[j]):
            map1[i] = n_key
            map2[j] = n_key 
            union_key.append(unique_key1[i].item())
            union_nkey1.append(nkey1[i].item())
            union_nkey2.append(nkey2[j].item())
            n_key = n_key + 1
            i = i + 1
            j = j + 1 
        elif(unique_key1[i] < unique_key2[j]):
            map1[i] = n_key
            union_key.append(unique_key1[i].item())
            union_nkey1.append(nkey1[i].item())
            union_nkey2.append(1)
            n_key = n_key + 1
            i = i + 1
        else:
            map2[j] = n_key
            union_key.append(unique_key2[j].item())
            union_nkey1.append(1)
            union_nkey2.append(nkey2[j].item())
            n_key = n_key + 1
            j = j + 1
    if(i >= len1):
        while(j < len2):
            map2[j] = n_key
            union_key.append(unique_key2[j].item())
            union_nkey1.append(1)
            union_nkey2.append(nkey2[j].item())
            n_key = n_key + 1
            j = j + 1
    elif(j >=len2):
        while(i < len1):
            map1[i] = n_key
            union_key.append(unique_key1[i].item())
            union_nkey1.append(nkey1[i].item())
            union_nkey2.append(1)
            n_key = n_key + 1
            i = i + 1
    inverse_idx1.apply_(lambda i: map1.get(i, -1))
    inverse_idx2.apply_(lambda i: map2.get(i, -1))
    union_key = torch.tensor(union_key)
    union_nkey1 = torch.tensor(union_nkey1)
    union_nkey2 = torch.tensor(union_nkey2)
    nkey_out = union_nkey1 * union_nkey2
    idx_out = torch.cat((torch.tensor([0]), torch.cumsum(nkey_out, dim=0)))
    res_table = TensorTable()
    res_table.column_name = col1 + col2
    out_tensor = torch.full([idx_out[-1], len(res_table.column_name)], float("nan"))
    nkey = torch.zeros_like(idx_out)
    ncol1 = len(col1)
    """
    for i in range(idx_out.shape[0] - 1):
        start_idx = idx_out[i] 
        end_idx = idx_out[i+1]
        out_tensor[start_idx:end_idx, 0:1] = union_key[i]
    """
    for i in range(tensor1.shape[0]):
        idx = inverse_idx1[i]
        start_idx = idx_out[idx] + nkey[idx]
        end_idx = start_idx + union_nkey2[idx]         
        out_tensor[start_idx:end_idx, :ncol1] = tensor1[i, :]
        nkey[idx] = nkey[idx] + union_nkey2[idx]
    nkey = torch.zeros_like(idx_out)
    for i in range(tensor2.shape[0]):
        idx = inverse_idx2[i]
        for j in range(union_nkey1[idx]):
            start_idx = idx_out[idx] + nkey[idx] + j * union_nkey2[idx]
            end_idx = start_idx + 1
            out_tensor[start_idx:end_idx, ncol1:] = tensor2[i, :]
        nkey[idx] = nkey[idx] + 1
    res_table.tensor = out_tensor
    return res_table

def left_join(table1, key1, table2, key2):
    """
    left join
    """
    col1, tensor1, key_tensor1 = parse_left_table(table1, key1)
    col2, tensor2, key_tensor2 = parse_right_table(table2, key2)
    unique_key1, inverse_idx1, nkey1 = torch.unique(key_tensor1, return_inverse=True, return_counts=True)
    unique_key2, inverse_idx2, nkey2 = torch.unique(key_tensor2, return_inverse=True, return_counts=True)
    if(unique_key1.type()!=unique_key2.type()):
        unique_key2 = unique_key2.type_as(unique_key1)
    print(unique_key1, unique_key2)
    print(inverse_idx1, inverse_idx2)
    left_key = unique_key1
    i = 0
    j = 0
    len1 = unique_key1.shape[0]
    len2 = unique_key2.shape[0]
    left_nkey1 = nkey1
    left_nkey2 = []
    map2 = {}
    while((i < len1) & (j < len2)):
        if(unique_key1[i] == unique_key2[j]):
            map2[j] = i 
            left_nkey2.append(nkey2[j].item())
            i = i + 1
            j = j + 1 
        elif(unique_key1[i] < unique_key2[j]):
            map2[j] = -1
            left_nkey2.append(1)
            i = i + 1
        else:
            j = j + 1
    if(j >=len2):
        while(i < len1):
            map2[i] = -1
            left_nkey2.append(1)
            i = i + 1
    inverse_idx2.apply_(lambda i: map2.get(i, -1))
    left_nkey2 = torch.tensor(left_nkey2)
    nkey_out = left_nkey1 * left_nkey2
    print(left_key)
    print(left_nkey1, left_nkey2)
    print(inverse_idx1, inverse_idx2)
    idx_out = torch.cat((torch.tensor([0]), torch.cumsum(nkey_out, dim=0)))
    res_table = TensorTable()
    res_table.column_name = col1 + col2
    out_tensor = torch.full([idx_out[-1], len(res_table.column_name)], float("nan"))
    nkey = torch.zeros_like(idx_out)
    ncol1 = len(col1)
    for i in range(tensor1.shape[0]):
        idx = inverse_idx1[i]
        start_idx = idx_out[idx] + nkey[idx]
        end_idx = start_idx + left_nkey2[idx]         
        out_tensor[start_idx:end_idx, :ncol1] = tensor1[i, :]
        nkey[idx] = nkey[idx] + left_nkey2[idx]
    nkey = torch.zeros_like(idx_out)
    for i in range(tensor2.shape[0]):
        idx = inverse_idx2[i]
        for j in range(left_nkey1[idx]):
            start_idx = idx_out[idx] + nkey[idx] + j * left_nkey2[idx]
            end_idx = start_idx + 1
            out_tensor[start_idx:end_idx, ncol1:] = tensor2[i, :]
        nkey[idx] = nkey[idx] + 1
    res_table.tensor = out_tensor
    return res_table

def right_join(table1, key1, table2, key2):
    """
    right join
    """
    col1, tensor1, key_tensor1 = parse_right_table(table1, key1)
    col2, tensor2, key_tensor2 = parse_right_table(table2, key2)
    unique_key1, inverse_idx1, nkey1 = torch.unique(key_tensor1, return_inverse=True, return_counts=True)
    unique_key2, inverse_idx2, nkey2 = torch.unique(key_tensor2, return_inverse=True, return_counts=True)
    if(unique_key1.type()!=unique_key2.type()):
        unique_key1 = unique_key1.type_as(unique_key2)
    print(unique_key1, unique_key2)
    print(inverse_idx1, inverse_idx2)
    right_key = unique_key2
    i = 0
    j = 0
    len1 = unique_key1.shape[0]
    len2 = unique_key2.shape[0]
    right_nkey1 = []
    right_nkey2 = nkey2
    map1 = {}
    while((i < len1) & (j < len2)):
        if(unique_key1[i] == unique_key2[j]):
            map1[i] = j 
            right_nkey1.append(nkey1[i].item())
            i = i + 1
            j = j + 1 
        elif(unique_key2[j] < unique_key1[i]):
            map1[i] = -1
            right_nkey1.append(1)
            j = j + 1
        else:
            i = i + 1
    if(i >=len1):
        while(j < len2):
            map1[j] = -1
            right_nkey1.append(1)
            j = j + 1
    inverse_idx1.apply_(lambda i: map1.get(i, -1))
    right_nkey1 = torch.tensor(right_nkey1)
    nkey_out = right_nkey1 * right_nkey2
    print(right_key)
    print(right_nkey1, right_nkey2)
    print(inverse_idx1, inverse_idx2)
    idx_out = torch.cat((torch.tensor([0]), torch.cumsum(nkey_out, dim=0)))
    res_table = TensorTable()
    res_table.column_name = [key2] + col1 + col2
    out_tensor = torch.full([idx_out[-1], len(res_table.column_name)], float("nan"))
    nkey = torch.zeros_like(idx_out)
    ncol1 = len(col1)
    for i in range(idx_out.shape[0] - 1):
        start_idx = idx_out[i] 
        end_idx = idx_out[i+1]
        out_tensor[start_idx:end_idx, 0:1] = right_key[i]
    for i in range(tensor1.shape[0]):
        idx = inverse_idx1[i]
        start_idx = idx_out[idx] + nkey[idx]
        end_idx = start_idx + right_nkey2[idx]         
        out_tensor[start_idx:end_idx, 1:ncol1+1] = tensor1[i, :]
        nkey[idx] = nkey[idx] + right_nkey2[idx]
    nkey = torch.zeros_like(idx_out)
    for i in range(tensor2.shape[0]):
        idx = inverse_idx2[i]
        for j in range(right_nkey1[idx]):
            start_idx = idx_out[idx] + nkey[idx] + j * right_nkey2[idx]
            end_idx = start_idx + 1
            out_tensor[start_idx:end_idx, ncol1+1:] = tensor2[i, :]
        nkey[idx] = nkey[idx] + 1
    res_table.tensor = out_tensor
    return res_table

def cross_join(table1, table2, use_multi_processing=True, nprocesses=None):
    """
    Cross join
    table1 : TensorTable
        left table to be joined
    table2: TensorTable
        right table to be joined
    use_multi_processing: bool
        If True, using multi processing in tensor copy
        if Flase, not using
    processes: int
        number of processes
    """
    res_table = TensorTable()
    out_len = table1.tensor.shape[0] * table2.tensor.shape[0]
    res_table.column_name = table1.column_name + table2.column_name
    col_len = len(res_table.column_name)
    out_tensor = torch.empty([out_len, col_len]) 
    copy = CrossJoinCopy(table1.tensor, table2.tensor, out_tensor, nprocesses)
    if(use_multi_processing == True):
        copy.multi_processing_copy()
    else:
        copy.single_copy()
    res_table.tensor = out_tensor
    return res_table 

def join(table1, key1, table2, key2, join_type="inner_join", use_multi_processing=True, nprocesses=None):
    """
    Join two tensor tables using key1 and key2
    join_type: join type, including inner_join, full_join, left_join, right_join, cross_join
    """
    if(join_type == "inner_join"):
        res_table = inner_join(table1, key1, table2, key2, use_multi_processing, nprocesses)
    elif(join_type == "full_join"):
        res_table = full_join(table1, key1, table2, key2)
    elif(join_type == "left_join"):
        res_table = left_join(table1, key1, table2, key2)
    elif(join_type == "right_join"):
        res_table = right_join(table1, key1, table2, key2)
    else:
        print("Unknown join type")
    return res_table

