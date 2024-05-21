"""Tree models, based on relay"""
import tvm
from tvm import relay
from collections import namedtuple

def tree_gemm(data, internal_node, leaf_node, dtype, sparse_replacing, dtype_converting):
    """
    S [internal_node, X_shape] The relationship between internal node and feature
    T [internal_node, 1] Threshold for each internal node
    B [leaf_node, internal_node] The relationship between lead node and internal node
    L [leaf_node,] Label for each leaf node
    """
    if(dtype_converting == True):
        min_dtype = "int8"
    else:
        min_dtype = dtype
    index_dtype = "int32"
    if(sparse_replacing in ["csr", "bsr", "csc"]):
        S_data = relay.var("S_data", dtype=dtype)
        S_indices = relay.var("S_indices", dtype=index_dtype)
        S_indptr = relay.var("S_indptr", dtype=index_dtype)
        Sparse = namedtuple("Sparse", ["data", "indices", "indptr"])
        S = Sparse(S_data, S_indices, S_indptr)
        if(sparse_replacing == "csc"):
            y = relay.nn.sparse_dense_csc(data, S, internal_node)
        else:
            y = relay.nn.sparse_dense(data, S)
    elif(sparse_replacing == "coo"):
        S_data = relay.var("S_data", dtype=dtype)
        S_row = relay.var("S_row", dtype=index_dtype)
        S_col = relay.var("S_col", dtype=index_dtype)
        Sparse = namedtuple("Sparse", ["data", "row", "col"])
        S = Sparse(S_data, S_row, S_col)
        y = relay.nn.sparse_dense_coo(data, S, internal_node)
    elif(sparse_replacing == "dia"):
        S_data = relay.var("S_data", dtype=dtype)
        S_offsets = relay.var("S_offsets", dtype=index_dtype)
        Sparse = namedtuple("Sparse", ["data", "offsets"])
        S = Sparse(S_data, S_offsets)
        y = relay.nn.sparse_dense_dia(data, S, internal_node)
    elif(sparse_replacing == "lil"):
        S_data = relay.var("S_data", dtype=dtype)
        S_rows = relay.var("S_rows", dtype=index_dtype)
        S_non_zero_cols = relay.var("S_non_zero_cols", dtype=index_dtype)
        Sparse = namedtuple("Sparse", ["data", "rows", "non_zero_cols"])
        S = Sparse(S_data, S_rows, S_non_zero_cols)
        y = relay.nn.sparse_dense_lil(data, S)
    else:
        y = relay.nn.dense(data, relay.var("S", dtype=dtype), units=internal_node)
    # y = tvm.relay.nn.bitserial_dense(data, relay.var("S", dtype=dtype), units=internal_node, weight_bits=1)
    # [batch_size, internal_node]
    y = relay.greater(y, relay.var("T", shape=(internal_node,), dtype=dtype))
    #y = tvm.relay.nn.bitserial_dense(y, relay.var("B", dtype="bool"), units=leaf_node, pack_dtype="uint8", out_dtype="uint8")
    y = relay.cast(y, min_dtype)
    y = relay.nn.dense(y, relay.var("B", dtype=min_dtype), units=leaf_node, out_dtype=min_dtype)
    # [batch_size, leaf_node]
    y = relay.argmax(y, axis=-1)
    # [batch_size,]
    l = relay.var("L", shape=(leaf_node,), dtype=dtype)
    y = relay.take(l, y)
    # [batch_size,]
    return y

def decision_tree_classifier(data_shape, internal_node, leaf_node, dtype, sparse_replacing, dtype_converting):
    """
    Decision tree classifier
    """
    data = relay.var("data", shape=data_shape, dtype=dtype)
    y = tree_gemm(data, internal_node, leaf_node, dtype, sparse_replacing, dtype_converting)
    return y

def extra_tree_classifier(data_shape, internal_node, leaf_node, label, dtype="float32"):
    """
    Extra tree classifier
    """
    return decision_tree_classifier(data_shape, internal_node, leaf_node, label, dtype)

def decision_tree_regressor(data_shape, internal_node, leaf_node, dtype, sparse_replacing, dtype_converting):
    """
    Decision tree regressor
    """
    data = relay.var("data", shape=data_shape, dtype=dtype)
    y = tree_gemm(data, internal_node, leaf_node, dtype, sparse_replacing, dtype_converting)
    return y

def extra_tree_regressor(data_shape, internal_node, leaf_node, label, dtype="float32"):
    """
    Extra tree regressor
    """
    return decision_tree_regressor(data_shape, internal_node, leaf_node, label, dtype)
