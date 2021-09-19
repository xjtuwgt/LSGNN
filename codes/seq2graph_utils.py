## definition of graph: DGL graph -> Heter Graph (node id, node type, edge id, edge type)
## order guided graph construction, but the local order (window size) can be shuffled
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from dgl.heterograph import DGLHeteroGraph
import torch
import dgl

def sliding_window_fast(seq_len: int, start_offset=0, window_size=24):
    """
    https://paperswithcode.com/method/sliding-window-attention
    :param seq_len:
    :param start_offset:
    :param window_size:
    :return:
    """
    assert window_size >=3
    sliding_seq_len = seq_len - start_offset
    if window_size >= sliding_seq_len:
        window_size = sliding_seq_len // 2 - 1
        window_size = window_size if window_size >= 1 else 1
    seq_np = np.arange(sliding_seq_len) + start_offset
    sliding_dst_array = sliding_window_view(x=seq_np, window_shape=window_size)
    row_n, col_n = sliding_dst_array.shape
    slide_last_idx = sliding_dst_array[row_n-1][0] + 1
    assert col_n == window_size
    src_array = sliding_dst_array[:,0].reshape(row_n, 1)
    sliding_src_array = np.repeat(src_array, col_n - 1, axis=1)
    sliding_src_array = sliding_src_array.flatten()
    sliding_dst_array = sliding_dst_array[:,1:].flatten()
    #####################################################################################
    diag_src_array = diag_dst_array = seq_np
    #####################################################################################
    pad_len = seq_len - slide_last_idx
    pad_src, pad_dst = np.triu_indices(pad_len, 1)
    pad_src = pad_src + slide_last_idx
    pad_dst = pad_dst + slide_last_idx
    # ###############################################################
    sliding_src = np.concatenate([sliding_src_array, sliding_dst_array, pad_src, pad_dst, diag_src_array])
    sliding_dst = np.concatenate([sliding_dst_array, sliding_src_array, pad_dst, pad_src, diag_dst_array])
    assert len(sliding_src) == len(sliding_dst)
    return sliding_src, sliding_dst

def sliding_window_with_position_fast(seq_len: int, start_offset=0, window_size=24):
    """
    adding relative position
    :param seq_len:
    :param start_offset:
    :param window_size:
    :return:
    """
    assert window_size >=3 and seq_len >=4
    sliding_seq_len = seq_len - start_offset
    if window_size >= sliding_seq_len:
        window_size = sliding_seq_len // 2 - 1
        window_size = window_size if window_size >= 1 else 1
    seq_np = np.arange(sliding_seq_len) + start_offset
    sliding_dst_array = sliding_window_view(x=seq_np, window_shape=window_size)
    row_n, col_n = sliding_dst_array.shape
    slide_last_idx = sliding_dst_array[row_n - 1][0] + 1
    assert col_n == window_size
    sliding_src_array = np.tile(sliding_dst_array[:, 0].reshape(row_n, 1), (1, col_n - 1))
    sliding_src_array = sliding_src_array.flatten()
    sliding_dst_array = sliding_dst_array[:,1:].flatten()
    #####################################################################################
    forward_pos_array = np.tile(np.arange(1, window_size), row_n)
    backward_pos_array = forward_pos_array + window_size
    #####################################################################################
    diag_src_array = diag_dst_array = seq_np
    diag_pos_array = np.zeros(sliding_seq_len, dtype=np.int32)
    #####################################################################################
    pad_len = seq_len - slide_last_idx
    pad_src, pad_dst = np.triu_indices(pad_len, 1)
    pad_src = pad_src + slide_last_idx
    pad_dst = pad_dst + slide_last_idx
    pad_forward = pad_dst - pad_src
    pad_backward = pad_forward + window_size
    #####################################################################################
    sliding_src = np.concatenate([sliding_src_array, sliding_dst_array, pad_src, pad_dst, diag_src_array])
    sliding_dst = np.concatenate([sliding_dst_array, sliding_src_array, pad_dst, pad_src, diag_dst_array])
    sliding_pos = np.concatenate([forward_pos_array, backward_pos_array, pad_forward, pad_backward, diag_pos_array])
    assert len(sliding_src) == len(sliding_dst) and len(sliding_dst) == len(sliding_pos)
    return sliding_src, sliding_dst, sliding_pos

def global_atten_edges(global_idx: list, seq_len: int):
    global_len = len(global_idx)
    global_src_array = np.tile(np.array(global_idx, dtype=int).reshape(global_len,1), (1, seq_len)).flatten()
    seq_dst_array = np.tile(np.arange(seq_len), global_len)
    assert len(global_src_array) == len(seq_dst_array)
    return global_src_array, seq_dst_array

def graph_triple_construction(seq_len: int, start_offset: int, window_size, global_idx: list, position: bool):
    if position:
        sliding_src, sliding_dst, sliding_position = sliding_window_with_position_fast(seq_len=seq_len, start_offset=start_offset,
                                                       window_size=window_size)
    else:
        sliding_src, sliding_dst = sliding_window_fast(seq_len=seq_len, start_offset=start_offset, window_size=window_size)
        sliding_len = len(sliding_src)
        sliding_position = np.full(sliding_len, 1, dtype=int)
    global_src, global_dst = global_atten_edges(global_idx=global_idx, seq_len=seq_len)
    # print(global_src, global_dst)
    global_len = len(global_src)
    src_list = [sliding_src, global_src, global_dst]
    dst_list = [sliding_dst, global_dst, global_src]
    if position:
        rel_list = [sliding_position, np.full(2*global_len, 2 * window_size, dtype=int)]
        relation_num = 2 * window_size + 1
    else:
        rel_list = [sliding_position, np.full(2 * global_len, 2, dtype=int)]
        relation_num = 3
    src = np.concatenate(src_list)
    dst = np.concatenate(dst_list)
    rel = np.concatenate(rel_list)
    assert len(src) == len(dst) and len(src) == len(rel)
    return src, dst, rel, relation_num

def seq2graph(sequence: list, global_idx: list, position, start_offset: int = 0, window_size=2) -> DGLHeteroGraph:
    """
    :param sequence:
    :param global_idx:
    :param position:
    :param start_offset:
    :param window_size:
    :return:
    """
    number_of_nodes = len(sequence)
    src, dst, rel, relation_num = graph_triple_construction(seq_len=number_of_nodes, global_idx=global_idx, window_size=window_size,
                                              start_offset=start_offset, position=position)
    # print(src)
    # print(dst)
    # print(rel)
    # temp_matrix = np.zeros((number_of_nodes, number_of_nodes))
    # for _ in range(len(src)):
    #     temp_matrix[src[_]][dst[_]] = rel[_]
    # print(temp_matrix)
    # print(len(rel), temp_matrix.shape)
    graph = dgl.graph(num_nodes=number_of_nodes, data=(src, dst))
    node_id = torch.arange(0, number_of_nodes, dtype=torch.long)
    node_type = torch.LongTensor(sequence)
    graph.ndata.update({'n_id': node_id, 'n_type': node_type})
    graph.edata['e_type'] = torch.from_numpy(rel)
    assert number_of_nodes == graph.number_of_nodes()
    return graph

if __name__ == '__main__':
    from time import time
    x = list(range(6))
    g = seq2graph(sequence=x, global_idx=list(range(2)), start_offset=2, window_size=3, position=True)
    print()
#     # x = np.arange(10)
#     # sliding_dst_array = sliding_window_view(x=x, window_shape=3)
#     # print(sliding_dst_array)
#     # # sliding_dst_array = sliding_window_view(x=x, window_shape=5)
#     # # print(sliding_dst_array)
#     x = list(range(9096))
    # sliding_dst_array = sliding_window_view(x=x, window_shape=6)
    # start_time = time()
    # for i in range(100):
    #     sliding_dst_array = sliding_window_view(x=x, window_shape=512)
    #     global_atten_edges(global_idx=list(range(200)), seq_len=9096)
    #     # g = seq2graph(sequence=x, global_idx=list(range(200)), start_offset=2, window_size=512, position=True)
    # print(time() - start_time)
        # print(i)
#     # # print(g.adjacency_matrix())
#     # print(g.number_of_edges())
#     start_time = time()
#     for i in range(100):
#         # x = np.tile(np.arange(10), 2)
#         sliding_window_with_position_fast_1(seq_len=10000, window_size=24)
#     print(time() - start_time)
#     # print(x)
    start_time = time()
    for i in range(1):
        # global_atten_edges_1(global_idx=list(range(200)), seq_len=10000)
        sliding_window_with_position_fast(seq_len=3000, window_size=24)
        # seq_idx_array = np.arange(10).reshape(1, 10).repeat(2, axis=0)
        # seq_idx_array = seq_idx_array.flatten()
    print(time() - start_time)
    # print(seq_idx_array)
