import numpy as np
import tensorflow as tf
from keras import Sequential
from keras import layers as Layers
from keras import ops
from keras import Model
from name_list import *
import keras
from netCDF4 import Dataset
import time
from random import sample
import helper_functions as hf

restart = True
gridsize = N

tf.config.run_functions_eagerly(True)

domain = [np.pi/18 * i for i in range(36)]
domain_ids = [i for i in range(36)]

points64 = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]

class GridNode:
    def __init__(self, u1, v1, h1, u2, v2, h2, St, lat_id, lon_id):
        self.u1 = u1
        self.v1 = v1
        self.h1 = h1
        self.u2 = u2
        self.v2 = v2
        self.h2 = h2
        self.St = St
        self.lat_id = lat_id
        self.lon_id = lon_id
        self.coslat = np.cos(domain[lat_id])
        self.sinlon = np.sin(domain[lon_id])
        self.coslon = np.cos(domain[lon_id])
        self.edgein = []
        self.edgeout = []

    def add_edgein(self, edge):
        self.edgein.append(edge)

    def add_edgeout(self, edge):
        self.edgeout.append(edge)

    def get_vector(self):
        self_x, self_y = ((2+self.coslat)*self.coslon), ((2+self.coslat)*self.sinlon)
        self_z = np.sqrt(1 - (2 - np.sqrt(self_x **2 + self_y ** 2))**2)
        self_vector = [self_x, self_y, self_z]
        return self_vector


class MeshNode:
    def __init__(self, lat_id, lon_id):
        self.lat_id = lat_id
        self.lon_id = lon_id
        self.coslat = np.cos(points64[lat_id])
        self.sinlon = np.sin(points64[lon_id])
        self.coslon = np.cos(points64[lon_id])
        self.edgein = []
        self.edgeout = []

    def add_edgein(self, edge):
        self.edgein.append(edge)

    def add_edgeout(self, edge):
        self.edgeout.append(edge)

    def get_vector(self):
        self_x, self_y = ((2+self.coslat)*self.coslon), ((2+self.coslat)*self.sinlon)
        self_z = np.sqrt(1 - (2 - np.sqrt(self_x **2 + self_y ** 2))**2)
        self_vector = [self_x, self_y, self_z]
        return self_vector


class Edge:
    def __init__(self, pointingto, pointingfrom):
        pointingto_vec = pointingto.get_vector()
        pointingfrom_vec = pointingfrom.get_vector()

        self.length = np.sqrt(np.array(pointingto_vec) ** 2 + np.array(pointingfrom_vec) ** 2)
        self.vectordifference = np.array(pointingto_vec) - np.array(pointingfrom_vec)
        self.pointingfrom = pointingfrom
        self.pointingto = pointingto
        self.pointingfrom.add_edgeout(self)
        self.pointingto.add_edgein(self)


class Graph:
    def __init__(self, grid_nodes, mesh_nodes4, mesh_nodes16, mesh_nodes64, mesh_edges4, mesh_edges16, mesh_edges64, gridtomesh_edges, meshtogrid_edges):
        self.grid_nodes = grid_nodes
        self.mesh_nodes4 = mesh_nodes4
        self.mesh_nodes16 = mesh_nodes16
        self.mesh_nodes64 = mesh_nodes64
        self.mesh_edges4 = mesh_edges4
        self.mesh_edges16 = mesh_edges16
        self.mesh_edges64 = mesh_edges64
        self.gridtomesh_edges = gridtomesh_edges
        self.meshtogrid_edges = meshtogrid_edges

    def describe(self):
        print("Number of grid nodes is " + str(len(self.grid_nodes)))
        print("Number of mesh nodes is " + str(len(self.mesh_nodes4) + len(self.mesh_nodes16) + len(self.mesh_nodes64)))
        print("Number of mesh edges is " + str(len(self.mesh_edges4) + len(self.mesh_edges16) + len(self.mesh_edges64)))
        print("Number of grid-to-mesh edges is " + str(len(self.gridtomesh_edges)))
        print("Number of mesh-to-grid edges is " + str(len(self.meshtogrid_edges)))



def node_length(node1, node2):
    vec1 = node1.get_vector()
    vec2 = node2.get_vector()

    length = np.sqrt(np.array(vec1) ** 2 + np.array(vec2) ** 2)

    return length[0]


def max_length(edge_list):
    length_list = []
    for i in edge_list:
        length_list.append(i.length)

    return np.max(length_list)


def build_grid(u1, v1, h1, u2, v2, h2, Wmat):
    grid_nodes = [GridNode(u1[j,k], v1[j,k], h1[j,k], u2[j,k], v2[j,k], h2[j,k], Wmat[j,k], j, k) for j in range(gridsize) for k in range(gridsize)]
    return grid_nodes


def build_mesh():
    mesh_nodes4 = [MeshNode(i*4,j*4) for i in range(2) for j in range(2)]
    mesh_nodes16 = [MeshNode(i*2,j*2) for i in range(3) for j in range(3)]
    mesh_nodes64 = [MeshNode(i,j) for i in range(4) for j in range(4)]
    return mesh_nodes4, mesh_nodes16, mesh_nodes64


def build_mesh_edges(mesh_nodes4, mesh_nodes16, mesh_nodes64):
    mesh_edges4 = []
    for i in mesh_nodes4:
        for j in mesh_nodes4:
            if i.lon_id == j.lon_id:
                if i.lat_id == 0 and j.lat_id == 4:
                    mesh_edges4.append(Edge(i, j))
                    mesh_edges4.append(Edge(j, i))
            if i.lat_id == j.lat_id:
                if i.lon_id == 0 and j.lon_id == 4:
                    mesh_edges4.append(Edge(i, j))
                    mesh_edges4.append(Edge(j, i))
    mesh_edges16 = []
    for i in mesh_nodes16:
        for j in mesh_nodes16:
            if i.lon_id == j.lon_id:
                if i.lat_id == j.lat_id+2 or (i.lat_id == 0 and j.lat_id == 6):
                    mesh_edges16.append(Edge(i, j))
                    mesh_edges16.append(Edge(j, i))
            if i.lat_id == j.lat_id:
                if i.lon_id == j.lon_id+2 or (i.lon_id == 0 and j.lon_id == 6):
                    mesh_edges16.append(Edge(i, j))
                    mesh_edges16.append(Edge(j, i))
    mesh_edges64 = []
    for i in mesh_nodes64:
        for j in mesh_nodes64:
            if i.lon_id == j.lon_id:
                if i.lat_id == j.lat_id+1 or (i.lat_id == 0 and j.lat_id == 7):
                    mesh_edges64.append(Edge(i, j))
                    mesh_edges64.append(Edge(j, i))
            if i.lat_id == j.lat_id:
                if i.lon_id == j.lon_id+1 or (i.lon_id == 0 and j.lon_id == 7):
                    mesh_edges64.append(Edge(i, j))
                    mesh_edges64.append(Edge(j, i))
    
    return mesh_edges4, mesh_edges16, mesh_edges64


def build_gridtomesh_edges(grid_nodes, mesh_nodes4, mesh_nodes16, mesh_nodes64, max_edge):
    gridtomesh_edges = []
    for i in grid_nodes:
        for j in mesh_nodes4:
            if node_length(i, j) <= 0.6 * max_edge:
                gridtomesh_edges.append(Edge(j, i))
    for i in grid_nodes:
        for j in mesh_nodes16:
            if node_length(i, j) <= 0.6 * max_edge:
                gridtomesh_edges.append(Edge(j, i))
    for i in grid_nodes:
        for j in mesh_nodes64:
            if node_length(i, j) <= 0.6 * max_edge:
                gridtomesh_edges.append(Edge(j, i))
    return gridtomesh_edges


def build_meshtogrid_edges(grid_nodes, mesh_nodes4, mesh_nodes16, mesh_nodes64):
    """
    unfinished
    """
    meshtogrid_edges = []
    for i in grid_nodes:
        for k in range(7):
            if k == 7:
                long = 7
            elif i.lon_id >= points64[k] and i.lon_id <= points64[k+1]:
                long = k
        for j in range(7):
            if k == 7:
                lat = 7
            elif i.lat_id >= points64[j] and i.lat_id <= points64[j+1]:
                lat = j
        for l in mesh_nodes64:
            if l.lon_id == long and l.lat_id == lat:
                meshtogrid_edges.append(Edge(i, l))
    return meshtogrid_edges





# 1 MLP for each gridtomesh edge 
# 1 MLP for each mesh node
# 1 MLP for each grid node


# 1 MLP for each mesh edge x number of layers
# 1 MLP for each mesh node x number of layers

# 1 MLP for each mesh to grid edge
# 1 MLP for each grid node 

# 1 MLP for each grid node





#class edge

def build_mlp(input_size, output_size):
    input = Layers.Input(shape=(1,input_size))
    hidden = Layers.Dense(100, activation="silu")(input)
    output = Layers.Dense(output_size, activation="linear")(hidden)
    model = Model(input,output)
    return model


def build_model(u1, v1, h1, u2, v2, h2, Wmat):
    grid_nodes = build_grid(u1, v1, h1, u2, v2, h2, Wmat)
    print("done")
    mesh_nodes4, mesh_nodes16, mesh_nodes64 = build_mesh()
    print("done")
    mesh_edges4, mesh_edges16, mesh_edges64 = build_mesh_edges(mesh_nodes4, mesh_nodes16, mesh_nodes64)
    print("done")
    max_edge = max_length(mesh_edges64)
    print("done")
    gridtomesh_edges = build_gridtomesh_edges(grid_nodes, mesh_nodes4, mesh_nodes16, mesh_nodes64, max_edge)
    print("done")
    meshtogrid_edges = build_meshtogrid_edges(grid_nodes, mesh_nodes4, mesh_nodes16, mesh_nodes64)
    print("done")
    graph = Graph(grid_nodes, mesh_nodes4, mesh_nodes16, mesh_nodes64, mesh_edges4, mesh_edges16, mesh_edges64, gridtomesh_edges, meshtogrid_edges)
    print("done")
    return graph

u1 = np.random.random((gridsize,gridsize))
v1 = np.random.random((gridsize,gridsize))
h1 = np.random.random((gridsize,gridsize))
u2 = np.random.random((gridsize,gridsize))
v2 = np.random.random((gridsize,gridsize))
h2 = np.random.random((gridsize,gridsize))
Wmat = np.random.random((gridsize,gridsize))
graph = build_model(u1, v1, h1, u2, v2, h2, Wmat)
graph.describe()




#def grid_node
#def mesh_node
#def mesh_edge
#def gridtomesh_edge
#def meshtogrid_edge
#def gridtomesh_edge_update(edge)

