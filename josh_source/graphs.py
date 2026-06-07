import numpy as np
from collections import defaultdict

from .tracker import Group

def _run_union_find(adjacency_matrix, cameras, instance_list, match_conflicts, cam_map, proj_cache):

        n_nodes = adjacency_matrix.shape[0]
        parent = {n: n for n in range(n_nodes)}
        cam_to_id = {c.name: i for i, c in enumerate(cameras)}
        camera_ids = [cam_to_id[tup[1]] for tup in instance_list]

        cameras_in_component = {n: {camera_ids[n]} for n in range(n_nodes)}


        def find(i):
            if parent[i] == i: return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j
                cameras_in_component[root_j].update(cameras_in_component[root_i])

        # TODO: put in the 'good' bad_edges back into the graph
        edge_conflicts = {frozenset(edge) for edge in match_conflicts}

        for i in range(n_nodes):
            for j in range(n_nodes):
                if adjacency_matrix[i][j] != np.inf and frozenset({i, j}) not in edge_conflicts:
                    union(i, j)

        # add in valid edges back in
        for u, v in edge_conflicts:
            root_u = find(u)
            root_v = find(v)

            if root_u == root_v:
                continue

            if cameras_in_component[root_u].isdisjoint(cameras_in_component[root_v]):
                # merge the two if valid subgraph
                parent[root_u] = root_v
                cameras_in_component[root_v].update(cameras_in_component[root_u])


        # find all groups
        distinct_groups = defaultdict(list)
        for node in range(n_nodes):
            root = find(node)
            distinct_groups[root].append(node)


        # check validity of groups
        invalid_instances = []
        groups = []
        for group in list(distinct_groups.values()):
            if len(group) < 2:
                continue
                invalid_instances.extend(group)
            elif Group.is_valid(group, instance_list):
                groups.append(Group(group, instance_list, cam_map, proj_cache))
            else:
                # print(f'Group {group} is invalid')
                # raise NotImplementedError
                return group
                groups.append(Group(group, instance_list, cam_map, proj_cache))
                # group = self._fix_group(group)

        return None

        # return groups



def _fix_group(group, adjacency_matrix, cameras, instance_list, match_conflicts, cam_map, proj_cache):
    
    parent = {n: n for n in group}
    cam_to_id = {cam.name: i for i, cam in enumerate(cameras)}
    camera_ids = [cam_to_id[t[1]]for t in instance_list]
    cameras_in_component = {n: {camera_ids[n]} for n in group}

    def find(i):
        if parent[i] == i: return i
        parent[i] =  find(parent[i])
        return parent[i]

    # extract all the possible edges
    edges = []
    n_nodes = len(group)
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            u, v = group[i], group[j]
            weight = adjacency_matrix[u, v]
            if weight != np.inf:
                edges.append((u, v, weight))

    edges.sort(key=lambda x: x[2])

    # return edges
    for u, v, _ in edges:
        root_u = find(u)
        root_v = find(v)

        # same parent --> do nothing; nodes already in same group
        if root_u == root_v:
            continue

        if cameras_in_component[root_v].isdisjoint(cameras_in_component[root_u]):
            # disjoint cameras --> valid to joint nodes
            parent[root_v] = root_u
            cameras_in_component[root_u].update(cameras_in_component[root_v])

        else:
            # non disjoint camera sets --> do not join nodes
            # don't connect the groups
            continue

    distinct_groups = defaultdict(list)
    for node in group:
        root = find(node)
        distinct_groups[root].append(node)
    
    return list(distinct_groups.values())