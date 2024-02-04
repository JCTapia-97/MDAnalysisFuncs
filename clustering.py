from __future__ import unicode_literals
from builtins import (zip, object)
import numpy as np


class Clustering(object):
    '''An object to manage clustering relationship.

    A cyclic linked list, child, is used to represent a
    cluster of nodes. Another array, root, is used for book
    keeping and root[i] points to the root node of the
    cluster of which node i is a member.

    size: number of nodes in the system.

    should_merge: a function that takes two indices and
    returns whether the two nodes belong to the same
    cluster.

    Whenever the node relationship changes, rebuild() can be
    used to re-compute clustering.

    If should_merge is provided when the constructor is
    called, rebuild() will be automatically invoked to
    compute initial state of clustering.

    If should_merge is not given, it is assumed that each
    node is initially in a cluster consisting of only
    itself. Subsequently, merge(i,j) can be used to merge
    two nodes.

    '''

    def __init__(self, size, should_merge=None):
        self.size = size

        if should_merge is None:
            self.reset()
        else:
            self.should_merge = should_merge
            self.rebuild(should_merge)

    def copy(self):
        c = self.__class__(self.size)

        c.child[:] = self.child
        c.root[:] = self.root
        c.num_cluster = self.num_cluster

        try:
            c.should_merge = self.should_merge
        except AttributeError:
            pass

        return c

    def delete_cluster(self, i, size=None):
        '''Delete the cluster containing node i.

        Return a reversely sorted array containing the
        deleted nodes.

        '''

        mol = self.get_cluster(i)
        mol.sort()
        mol = mol[::-1]
        mol_lowest_idx = np.amin(mol)
        if size is None:
            size = self.size
        child = self.child
        root = self.root
        self.num_cluster -= 1

        for j, (c, r) in enumerate(zip(child[:size], root[:size])):
            if c > mol_lowest_idx:
                child[j] -= (mol < c).sum()

            if r > mol_lowest_idx:
                root[j] -= (mol < r).sum()

        # mol is sorted large to small
        for j in mol:
            for k, (c, r) in enumerate(zip(child[j+1:size], root[j+1:size])):
                child[j+k] = c
                root[j+k] = r

            size -= 1

        orig_size = size + mol.size
        child[size:orig_size] = np.arange(size, orig_size)
        root[size:orig_size] = child[size:orig_size]

        return mol

    def get_all_clusters(self, size=None):
        '''Return all clusters in the system.

        Return a list, each element of which is an array of
        nodes that belong to the same cluster.

        '''

        if size is None:
            size = self.size

        clusters = []

        for i, i_root in enumerate(self.root[:size]):
            if i == i_root:
                clusters.append(self.get_cluster(i))

        return clusters

    def get_cluster(self, m):
        '''Return the cluster of which m is a member.

        Return an array containing all nodes belonging to
        the same cluster as m.

        '''

        child = self.child
        m_cluster = np.empty(self.size, dtype=np.int32)

        i = 0
        m_cluster[0] = m
        walker = child[m]
        while walker != m:
            i += 1
            m_cluster[i] = walker
            walker = child[walker]

        return m_cluster[:i+1]

    def in_same_cluster(self, i, j):
        '''Return whether nodes i and j are in the same cluster.'''

        if self.root[i] == self.root[j]:
            return True
        else:
            return False

    def merge(self, i, j):
        '''Merge two clusters of which nodes i and j are members.'''

        root = self.root

        i_root = root[i]
        j_root = root[j]

        if i_root == j_root:  # already in the same cluster
            return

        child = self.child

        # reroute all nodes in cluster j to point to i_root
        walker = j
        while True:
            root[walker] = i_root
            walker = child[walker]
            if walker == j:
                break

        # break the loop representing cluster j and insert
        # it after i
        j_child = child[j]
        child[j] = child[i]
        child[i] = j_child

        self.num_cluster -= 1

    def merge_all(self, l):
        '''Merge all clusters containing members from list l.'''

        i = l[0]
        for j in l[1:]:
            self.merge(i, j)

    def rebuild(self, should_merge=None, size=None):
        '''Re-compute the state of clustering.

        Return the number of distinct clusters identified.

        '''

        if should_merge is None:
            should_merge = self.should_merge
        if size is None:
            size = self.size
        self.reset()

        for i, i_child in enumerate(self.child[:size]):
            # i is not part of a cluster of which k (<i)
            # belongs. Therefore consider if i forms a
            # cluster with any following nodes.
            if i == i_child:
                walker = i
                while True:
                    for j, j_child in enumerate(self.child[i+1:size]):
                        # j is not part of clusters already
                        # found
                        if j+i+1 == j_child and should_merge(walker, j_child):
                            self.merge(walker, j_child)

                    walker = self.child[walker]
                    if walker == i:
                        break

        return self.num_cluster

    def reset(self):
        '''Reset clustering state.'''

        self.child = np.arange(self.size, dtype=np.int32)
        self.root = np.arange(self.size, dtype=np.int32)
        self.num_cluster = self.size
