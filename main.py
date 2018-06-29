# -*- coding: utf-8 -*-
#!/usr/bin/python3
# TODO build_cluster and clustering doubled...


import networkx
import sys
import timeit


def read_edges (file):
    edges = []
    with open(file) as input:
        for line in input:
            if line.split()[0] != '%':
                edges.append(line[:-1])
    return edges


# Prints a list of edges.
def print_edges (edges_list):
    for edge in edges_list:
        print(edge)


# Computes βs.
def betas (dmin, dmax, epsilon):
    n     = (1+epsilon)
    betas = []
    while n < dmax:
        if n > dmin:
            betas.append(n)
        n = n * (1+epsilon)
    return betas


# Builds a graph from a list of edges.
def init_graph (edges_list):
    return networkx.parse_edgelist(edges_list, nodetype = int,
            data=(('weight',int),('timestamp', int)))


# Retrieves the neighbours of node in the graph G.
def neigh (G, node):
    return G[node]


# Adds the edge edges_list[i] to the graph G.
def add_edge_str (G, edges_str, i):
    edges_list = edges_str[i].split()
    source = int(edges_list[0])
    destination = int(edges_list[1])
    w = int(edges_list[2])
    t = int(edges_list[3])
    G.add_edge(source, destination, weight=w, timestamp=t)


# Remove the edge edges_list[i] from the graph G.
def remove_edge_str (G, edges_str, i):
    edges_list = edges_str[i].split()
    source = int(edges_list[0])
    destination = int(edges_list[1])
    G.remove_edge(source, destination)
    if source != destination:
    # Avoids removing twice the same node, thus throwing an error upon removing
    # a node with a self loop.
        if G.degree(source) == 0:
            G.remove_node(source)
        if G.degree(destination) == 0:
            G.remove_node(destination)
    else:
        if G.degree(source) == 0:
            G.remove_node(source)


# Performs sliding window on a stream of edges. Adds a new one, removes the
# oldest, based on the stream edges_list.
def slide_graph_str (G, edges_list, window, i):
    if window+i < len(edges_list):
        remove_edge_str(G, edges_list, i)
        add_edge_str(G, edges_list, window+i)
    else:
         print("Index out of bound for edges_list", file=sys.stderr)


# Computes the distance from node to the closest center, given it has not been
# computed before.
def get_dist (G, node, d):
    # Twice as long: min([neigh(G, 1)[n]["weight"] for n in neigh(G, 1)])
    neigh_list  = neigh(G, node)
    return min([neigh_list[n]["weight"]+d[n] for n in neigh_list if n in d])


# Computes the distance of all the unclustered nodes to the new center.
# Used for initializing.
def gen_dist (G, unclustered, center, d):
    if center not in G.nodes():
        print(center, "is not a node of the current graph", file=sys.stderr)
        return d
    #print("New center is", start)
    visited, queue = set(), [center]
    while queue:
        node = queue.pop(0)
        if node not in visited:
            if node == center:
                d[node] = 0
            else:
                if node in d:
                    print(node, "is already in d:", d[node], file=sys.stderr)
                d[node] = get_dist(G, node, d)
                #print("Distance from", node, "to", start,  "is", d[node])
            visited.add(node)
            neigh_set = set(i for i in neigh(G, node) if i in unclustered)
            queue.extend(neigh_set - visited)
    return d


# Builds a cluster within 2β distance of a given center.
def build_cluster (unclustered, beta, center, d):
    cluster = set()
    cluster.add(center)
    for node in unclustered:
        if node in d and d[node] < 2 * beta:
            cluster.add(node)
    return cluster


# Initial cluster.
# Build clusters out of random centers.
def clustering (G, k, betas):

    result = []
    nodes_list = G.nodes()
    for beta in betas:
        centers     = set()
        clusters    = []
        unclustered = set(nodes_list)
        nb_centers  = 0
        d = {}

        print(beta, "\t", betas.index(beta)+1, "out of\t", len(betas),
                flush=True)
        while nb_centers < k and len(unclustered):

            # Random here.
            new_center = unclustered.pop()
            d = gen_dist(G, unclustered, new_center, d)
            centers.add(new_center)
            clusters.append(build_cluster(unclustered, beta, new_center, d))
            unclustered = set(x for x in unclustered if x not in clusters[-1])
            nb_centers += 1

        result.append([centers,clusters,unclustered,beta,d])
    return result


# Retrieves the minimum edge weight over all the edges in the graph. Here
# called dmin.
def get_min_weight(G):
    return min([e[2]["weight"] for e in G.edges(data = True)])


# Updates the distance from node to its closest center. Returns said distance.
def update_dist (G, node, d):
    #neigh_list  = neigh(G, node)
    #weight_list = [neigh_list[n]["weight"]+d[n] for n in neigh_list if n in d]
    #return min(min(weight_list),d[node])
    return min(get_dist(G, node, d),d[node])


# Retrieves the cluster in which node is.
def get_cluster (L, node):
    for cluster in L[1]:
        if node in cluster:
            return cluster


# Sorts a set of nodes into a list of nodes according to the distance to the
# center of the cluster.
def sort (affected, d):
    return sorted(affected, key=lambda k: d[k])


# Updates the distance from node to the closest center. Returns the distance
# along with the neighbour on the shortest path to said center.
def update_dist_neigh (G, node, d):
    neigh_list  = neigh(G, node)
    #print("This is the list of neighbours of", node, neigh_list)
    weight_list = [(n,neigh_list[n]["weight"]+d[n]) for n in neigh_list if n in d]
    #print("This is the weight_list", weight_list)
    if not weight_list:
        return (-1,-1)
    else:
        return min(weight_list, key = lambda t: t[1])


# Insterts an edge in an existing cluster if possible, otherwise as a new
# center, otherwise as an unclustered point.
def edge_insertion (G, L, k, edge_str):

    edge_list   = edge_str.split()
    source      = int(edge_list[0])
    destination = int(edge_list[1])
    weight      = int(edge_list[2])
    timestamp   = int(edge_list[3])

    print("Adding: Source", source, "Destination", destination, timestamp)

    if source not in G.nodes() and destination not in G.nodes():
    # Adding a new connected componend with two nodes.
        print(source, "not in the graph.", destination, "not in the graph.")
        add_edge_str(G, [edge_str], 0)
        #print(G.edges())
        #print(G.nodes())
        for i in range(len(L)):
            centers     = L[i][0]
            clusters    = L[i][1]
            unclustered = L[i][2]
            beta        = L[i][3]
            d           = L[i][4]
            #print(L[i])

            if len(centers) < k and weight < 2*beta:
            # Tries to put them both in a new cluster.
            # COULD BE RANDOM.
                centers.add(destination)
                clusters.append({source, destination})
                d[destination] = 0
                d[source] = weight
            elif len(centers) < k-1 and weight >= 2*beta:
            # If they're too far apart, create one cluster each.
                centers.add(destination)
                clusters.append({destination})
                d[destination] = 0
                centers.add(source)
                clusters.append({source})
                d[source] = 0
            elif len(centers) == k-1 and weight >= 2*beta:
            # COULD BE RANDOM.
                centers.add(source)
                d[source] = 0
                clusters.append({source})
                unclustered.add(destination)
            elif len(centers) == k:# and weight >= 2*beta:
            # Otherwise, they're unclustered.
                #unclustered = unclustered.union({source, destination})

                L[i] = (centers, clusters, unclustered.union({source, destination}), beta, d)
            else:
                print("UNMATCHED CASE under", source, "and", destination, "both in the graph.", file=sys.stderr)
                #print("Number of centers", len(centers), "k", k, "2β", 2*beta, "weight", weight)
    elif source in G.nodes() and destination not in G.nodes():
    # Adds a new edge along with a new node to the graph.
    # Makes sense only when the graph is undirected.
        print(source, "in the graph.", destination, "not in the graph.")
        add_edge_str(G, [edge_str], 0)
        #print(G.edges())
        #print(G.nodes())
        for i in range(len(L)):
            centers     = L[i][0]
            clusters    = L[i][1]
            unclustered = L[i][2]
            beta        = L[i][3]
            d           = L[i][4]
            #print(L[i])

            if source in unclustered:
                unclustered.add(destination)
            elif d[source]+weight < 2*beta:
            # The source is clustered and the destination is within 2β distnce.
                d[destination] = d[source]+weight
                get_cluster(L[i], source).add(destination)
            elif d[source]+weight >= 2*beta and len(centers) < k:
            # The destination isn't within clustering distance but can be added
            # as a new center.
                centers.add(destination)
                clusters.append({destination})
                d[destination] = 0
            elif d[source]+weight >= 2*beta and len(centers) >= k:
            # The destination isn't within clustering distance and there are
            # already k centers. >=k has been written out of robustness but
            # shouldn't happen.
                unclustered.add(destination)
            else:
                print("UNMATCHED CASE under", source, "in the graph and", destination, "not in the graph.", file=sys.stderr)
                #print("Number of centers", len(centers), "k", k, "2β", 2*beta, "weight", weight)
    elif source not in G.nodes() and destination in G.nodes():
        # Symetrical
        print(source, "not in the graph.", destination, "in the graph.")
        #print(G.nodes())
        #print(G.edges())
        add_edge_str(G, [edge_str], 0)
        for i in range(len(L)):
            centers     = L[i][0]
            clusters    = L[i][1]
            unclustered = L[i][2]
            beta        = L[i][3]
            d           = L[i][4]
            #print(L[i])

            if destination in unclustered:
                unclustered.add(source)
            elif d[destination]+weight < 2*beta:
                d[source] = d[destination]+weight
                get_cluster(L[i], destination).add(source)
            elif d[destination]+weight >= 2*beta and len(centers) < k:
                centers.add(source)
                clusters.append({source})
                d[source] = 0
            elif d[destination]+weight >= 2*beta and len(centers) >= k:
                unclustered.add(source)
            else:
                print("UNMATCHED CASE under", source, "not in the graph and", destination, "in the graph.", file=sys.stderr)
                #print("Number of centers", len(centers), "k", k, "2β", 2*beta, "weight", weight)
    elif source in G.nodes() and destination in G.nodes() and not(G.has_edge(source, destination) and weight == G.get_edge_data(source,destination)["weight"]):
    # Adding an edge between two existing nodes.
    # Chose to not add an edge if only the timestamp changes.
        print(source, "in the graph.", destination, "in the graph.")
        add_edge_str(G, [edge_str], 0)
        #print(G.edges())
        #print(G.nodes())
        for i in range(len(L)):
            centers     = L[i][0]
            clusters    = L[i][1]
            unclustered = L[i][2]
            beta        = L[i][3]
            d           = L[i][4]
            #print(L[i])

            if source in unclustered and destination in unclustered:
            # Nothing to be done if they are both unclustered. This won't
            # create a shorter path towards any center.
                pass
            elif source in unclustered and destination not in unclustered:
            # One end of the new edge is clustered, the other isn't.
                if d[destination]+weight < 2*beta:
                # The destination is clustered and the source is within
                # clustering distance.
                    d[source] = d[destination]+weight
                    get_cluster(L[i], destination).add(source)
                    unclustered.remove(source)
                elif len(centers) < k:
                    centers.add(source)
                    clusters.append({source})
                    unclustered.remove(source)
                    d[source] = 0
                # Not needed?
                else:
                    unclustered.add(source)
            elif source not in unclustered and destination in unclustered:
            # Only makes sense when the graph is undirected.
                if d[source]+weight < 2*beta:
                    d[destination] = d[source]+weight
                    get_cluster(L[i], source).add(destination)
                    unclustered.remove(destination)
                elif len(centers) < k:
                    centers.add(destination)
                    clusters.append({destination})
                    unclustered.remove(destination)
                    d[destination] = 0
                else:
                    unclustered.add(destination)
            elif get_cluster(L[i], source) != get_cluster(L[i], destination):
            # Recluster affected of get_cluter(source) and affected of
            # get_cluter(destination)

                L[i], affected = get_affected_diff_clusters(L[i], source, destination, weight)
                # Recluster the affected.
                recluster(G, affected, L[i])
            elif get_cluster(L[i], source) == get_cluster(L[i], destination):
            # Adding an edge within an cluster.
                cluster = get_cluster(L[i], source)
                if cluster == get_cluster(L[i], destination) and d[source] != d[destination] and not(weight >= abs(d[source] - d[destination])):
                    # Get affected within the cluster.
                    L[i], affected = get_affected(L[i], cluster, source, destination, 0, 0)
                    recluster(G, affected, L[i])
                else:
                    print("This edge won't be on a shortest path since there exist a shorter path from", source, "to", destination, "(beta", beta, ")")
            else:
                print("UNMATCHED CASE under", source, "in the graph and", destination, "in the graph.", file=sys.stderr)
                #print("Number of centers", len(centers), "k", k, "2β", 2*beta, "weight", weight)

            L[i] = (centers, clusters, unclustered, beta, d)
    return L


# Retrieves the list of affected nodes when source and destination are in
# different clusters.
def get_affected_diff_clusters(L, source, destination, weight):
    centers     = L[0]
    clusters    = L[1]
    unclustered = L[2]
    beta        = L[3]
    d           = L[4]
    affected    = set()
    if weight+d[destination] < d[source]:
        cluster = get_cluster(L, source)
        for node in cluster:
            if d[node] > d[source]:
                affected.add(node)
            affected.add(source)
        # Removes the affected from the center. O(k)...
        #for j in range(len(clusters)):
        #    if clusters[j] == cluster:
        #        clusters[j] = cluster.difference(affected)
        #        centers = centers.difference(affected)
    if weight+d[source] < d[destination]:
        cluster = get_cluster(L, destination)
        for node in cluster:
            if d[node] > d[destination]:
                affected.add(node)
            affected.add(destination)
        # Removes the affected from the center. O(k)...
        #for j in range(len(clusters)):
        #    if clusters[j] == cluster:
        #        clusters[j] = cluster.difference(affected)
        #        centers = centers.difference(affected)
    # ICI
    L = remove_list_from(affected, L)
    #L = (centers, clusters, unclustered, beta, d)
    return L,affected


# Cleans L from nodes_list. Done after computing the affected nodes.
def remove_list_from(nodes_list, L):
    for node in nodes_list:
        if node in L[0]:
            L[0].remove(node)
        for c in L[1]:
            if node in c:
                c.remove(node)
                if c == set():
                    L[1].remove(c)
        if node in L[2]:
            L[2].remove(node)
    return L


# When a node is removed from the graph, removes all occurences of said node in
# L.
def remove_from(L, node):
    if node in L[0]:
        L[0].remove(node)
    for c in L[1]:
        if node in c:
            c.remove(node)
            if c == set():
                L[1].remove(c)
    if node in L[2]:
        L[2].remove(node)
    if node in L[4]:
        del L[4][node]
    return L


# Retrieves the affected nodes after an update of the edge (source,destination)
def get_affected(L, cluster, source, destination, source_deg, destination_deg):
    d = L[4]
    if d[source] > d[destination]:
        furthest = source
    else:
        furthest = destination
    affected = set()
    for node in cluster:
        if d[node] > d[furthest]:
            affected.add(node)
        # Needs to recluster the end furthest away from the center too.
        affected.add(furthest)
    if source_deg == 1:
        print("Removing", source)
        #print(L)
        L = remove_from(L, source)
        #print(L)
        #print("Removed", source)
        if source == furthest:
            affected.remove(source)
    if destination_deg == 1:
        print("Removing", destination)
        L = remove_from(L, destination)
        #print("Removed", destination)
        if destination == furthest:
            affected.remove(destination)
    L = remove_list_from(affected, L)
    return L,affected


# Deletes an edge. Only the nodes that could be affected by that edge.
# Deleting an edge between two unclustered nodes can not change the clusters
# thus no need to recluster anything.
# Deleting an edge between a clustered node and an unclustered one can't change
# the clusters since that edge was not on a shortest path.
# Deleting an edge between two clustered nodes that aren't in the same cluster
# won't change anyhing because if that edge was on a shortest path to a center,
# then both its ends would be in the same center.
def edge_deletion (G, L, k, edge_str):

    edge_list = edge_str.split()
    source = int(edge_list[1])
    destination = int(edge_list[0])
    w = int(edge_list[2])
    t = int(edge_list[3])

    print("Removing: Source", source, "Destination", destination)

    if source in G.nodes() and destination in G.nodes() and G.has_edge(source, destination):
        # Doesn't make sense to remove something that isn't in the graph.

        print(source, "in the graph.", destination, "in the graph")
        source_deg      = G.degree(source)
        destination_deg = G.degree(destination)
        remove_edge_str(G, [edge_str], 0)
        #print("Graph edges:", G.edges())
        #print("Graph nodes:", G.nodes())
        for i in range(len(L)):
            centers     = L[i][0]
            clusters    = L[i][1]
            unclustered = L[i][2]
            beta        = L[i][3]
            d           = L[i][4]
            #print("Level: ", L[i])

            # Nodes could lie in a shortest path only if both ends are within
            # the same cluster.
            cluster = get_cluster(L[i], source)
            if source not in unclustered and destination not in unclustered and cluster == get_cluster(L[i], destination) and d[source] != d[destination]:

                # Get affected within the cluster.
                L[i], affected = get_affected(L[i], cluster, source, destination, source_deg, destination_deg)
                # Recluster the affected nodes.
                recluster(G, affected, L[i])
                L[i] = (centers, clusters, unclustered, beta, d)
            else:
                #Edge isn't used anyway if doesn't lie in a unique cluster.
                print("It looks like this edge wasn't on a shortest path. (beta", beta,")")
                L[i] = clean_after_deletion_noaffected(L[i], source, destination, source_deg, destination_deg)
                check_inconsistencies(G, source, destination, L[i])
    else:
        print("Trying to remove an edge dont one of its ends has been removed. Requested the deletion of (", source, ",", destination, ") in", G.edges(), G.nodes(), file=sys.stderr)

    return L


# Checks for inconsistencies such as a node being in a level and not in the
# graph or vice versa.
def check_inconsistencies(G, source, destination, L):
    if source in G.nodes() and not in_level(source, L):
        print(source, "in", G.nodes(), "but not in", L, file=sys.stderr)
        sys.exit()
    if source not in G.nodes() and in_level(source, L):
        print(source, "not in", G.nodes(), "but in", L, file=sys.stderr)
        sys.exit()
    if destination in G.nodes() and not in_level(destination, L):
        print(destination, "in", G.nodes(), "but not in", L, file=sys.stderr)
        sys.exit()
    if destination not in G.nodes() and in_level(destination, L):
        print(destination, "not in", G.nodes(), "but in", L, file=sys.stderr)
        sys.exit()


# Needs to clean the level after deleting an edge that isn't on a shortest
# path.
def clean_after_deletion_noaffected (L, source, destination, source_deg, destination_deg):
    if source != destination:
        if source_deg == 1:
            print("Removing", source)
            L = remove_from(L, source)
        if destination_deg == 1:
            print("Removing", destination)
            L = remove_from(L, destination)
    else:
        if source_deg == 2:
            print("Removing", source)
            L = remove_from(L, source)
    return L


# Checks if node is in the level L.
def in_level (node, L):
    tmp_set = set()
    for c in L[1]:
        tmp_set = tmp_set.union(c)
    #print(node, "In centers?", node in L[0])
    #print("In clusters?", node in tmp_set)
    #print("In unclustered?", node in L[2])
    #print("Thus returning", (node in L[0] or node in tmp_set or node in L[2]))
    return (node in L[0] or node in tmp_set or node in L[2])


# Reclusters every node in node_set, following a change in the graph.
# A node can switch clusters.
def recluster (G, nodes_set, L):
    centers     = L[0]
    clusters    = L[1]
    unclustered = L[2]
    beta        = L[3]
    d           = L[4]
    nb_centers  = len(L[0])

    #print("Order of nodes by distance to center", sort(nodes_set, L[4]))
    sorted_nodes = sort(nodes_set, L[4])
    #Cleanup d from nodes to be reclustered.
    for node in nodes_set:
        if node in d:
            del d[node]
    #print(sorted_nodes)
    if len(sorted_nodes):
        for n in nodes_set:
            remove_from(L, n)
            #del L[4][n]
        for n in sorted_nodes:
            (neighbour, distance) = update_dist_neigh(G, n, L[4])
            if neighbour == -1 and distance == -1:
                #print("The node", n, "has no distance to any existing center")
                if len(L[0]) < k:
                    #print("Since there are <k centers, it is a new center COULD BE RANDOM")
                    centers.add(n)
                    clusters.append({n})
                    d[n] = 0
                else:
                    #print("Can't create a new center. Adding to unclustered")
                    L[2].add(n)
            else:
                #print("Node", n, "has distance", distance,
                #        "to its center and neighbour", neighbour)
                if distance < 2*L[3]:
                    #print("This is the cluster of ", n, ":", get_cluster(L, neighbour))
                    #print(n, "is at distance", distance, "of its center, path goes immediately through", neighbour)
                    L[4][n] = distance
                    get_cluster(L, neighbour).add(n)
                elif len(L[0]) < k:
                    #print("The node", node, "is too far from any other center")
                    #print("Adding as a new center")
                    centers.add(n)
                    clusters.append({n})
                    d[n] = 0
                else:
                    #print("The node", node, "is too far from any other center")
                    #print("Adding to unclustered bc >=k clusters")
                    L[2].add(n)


if __name__ == "__main__":

    if len(sys.argv) != 5:
        print("[*] python3 main.py k epsilon window file")
        sys.exit()

    k       = int(sys.argv[1])
    epsilon = float(sys.argv[2])
    window  = int(sys.argv[3])
    gfile   = sys.argv[4]

    # Parses.
    edges_str = read_edges(gfile)

    # Initialises the graph.
    G = init_graph(edges_str[:window])
    dmin = get_min_weight(G)
    dmax = networkx.diameter(max(networkx.connected_component_subgraphs(G),
                                key=len))

    # Initialises the clusters.
    res = clustering(G, k, betas(dmin, dmax, epsilon))

    # Sliding window.
    for i in range(len(edges_str)-window):
        edge_insertion(G, res, k, edges_str[window+i])
        edge_deletion(G, res, k, edges_str[i])
    #    slide_graph_str(G, edges_str, window, i)
    #    print(G.nodes())


    # Timer.
    # checkpoint = timeit.default_timer()
    # print("[*] It took:\t\t", timeit.default_timer() - checkpoint)

