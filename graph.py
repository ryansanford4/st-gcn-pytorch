import numpy as np
import torch

class Graph():
    """ The Graph to model the skeletons extracted by the openpose
    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).
        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D
        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points
    """

    def __init__(self,
                 bbox_in,
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.edge, self.num_node, self.batch_size = get_neighbour_link(bbox_in)
        self.center = 1
        # self.get_edge(neighbor_link)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, self.batch_size, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.batch_size, self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.batch_size, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.batch_size, self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                              center] > self.hop_dis[i, self.
                                                                     center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")

def get_neighbour_link(boxes_in, threshold=0.2):
    boxes_in = boxes_in.cpu()
    graph_boxes_positions = np.reshape(boxes_in, (-1, 4))
    graph_boxes_positions[:, 0] = (graph_boxes_positions[:, 0] + graph_boxes_positions[:, 2]) / 2
    graph_boxes_positions[:, 1] = (graph_boxes_positions[:, 1] + graph_boxes_positions[:, 3]) / 2
    graph_boxes_positions = graph_boxes_positions[:, :2].reshape(boxes_in.size()[0], -1, boxes_in.size()[2], 2)  # B, T, N, 2
    # Get average bbox position.
    graph_boxes_positions = torch.mean(graph_boxes_positions, dim=1)
    graph_boxes_distances = calc_pairwise_distance_3d(graph_boxes_positions, graph_boxes_positions)  # B, N, N
    # Get the neighbouring link
    neighbor_link = []
    for batch in graph_boxes_distances:
        ind = np.where(batch<threshold)
        batch_neighbor_link = []
        for x,y in zip(ind[0], ind[1]):
            if x >= y:
                batch_neighbor_link.append((x,y))
        neighbor_link.append(batch_neighbor_link)
    
    return neighbor_link, boxes_in.size()[2], boxes_in.size()[0]

def calc_pairwise_distance_3d(X, Y):
    """
    computes pairwise distance between each element
    Args: 
        X: [B,N,D]
        Y: [B,M,D]
    Returns:
        dist: [B,N,M] matrix of euclidean distances
    """
    B=X.shape[0]
    
    rx=X.pow(2).sum(dim=2).reshape((B,-1,1))
    ry=Y.pow(2).sum(dim=2).reshape((B,-1,1))
    
    dist=rx-2.0*X.matmul(Y.transpose(1,2))+ry.transpose(1,2)
    
    return torch.sqrt(dist)


def get_hop_distance(num_node, edge, batch_size, max_hop=1):
    A = np.zeros((batch_size, num_node, num_node))
    for b in range(batch_size):
        for i, j in edge[b]:
            A[b, j, i] = 1
            A[b, i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((batch_size, num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis

def normalize_digraph(A):
    Dl = np.sum(A, 1)
    batch_size = A.shape[0]
    num_node = A.shape[1]
    Dn = np.zeros((batch_size, num_node, num_node))
    AD = np.zeros((batch_size, num_node, num_node))
    for j in range(batch_size):
        for i in range(num_node):
            if Dl[j, i] > 0:
                Dn[j, i, i] = Dl[j, i]**(-1)
        AD[j] = np.dot(A[j], Dn[j])
    return AD

def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD