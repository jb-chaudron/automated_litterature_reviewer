from litterature_review import ArtificialLibraire, search_papers


swan = ArtificialLibraire()
swan.doc_from_research("feature+genre+text",
                       nb_paper=50)

swan.get_graph(core=True)
swan.update_core_paper(0.1,2)
swan.get_graph(core=True)
swan.update_core_paper(0.2,0.2)

swan.abstract2doc()
swan.add_weight_to_graph()
swan.get_communities()

import numpy as np
from scipy.spatial.distance import squareform,pdist
from scipy.cluster.hierarchy import linkage 
doc_vec = [doc.vector for (node,doc) in swan.docs.items()]
#dist_mat = squareform(pdist(doc_vec,metric="cosine"))
Z = linkage(doc_vec,metric="cosine")

from scipy.cluster.hierarchy import to_tree
node_init, list_nodes = to_tree(Z,rd=True)

def linkage_to_hierarchy(Z):

    threshold_max = np.array(sorted(Z[:,2],reverse=True)[:5])
    clust_list = []

    cutting_points = np.where(Z[:,2] >= min(threshold_max))[0]
    # On passe en revue toute la chaîne de production
    for i in range(Z.shape[0]):
        return None

def tree_to_hierarchical_cluster(node, count, cluster_list = []):
    left, right = node.left, node.right

    if ((min(left.count, right.count)/max(left.count,right.count)) > 0.2) and (min(left.count,right.count) > 1):
        cluster_list += [left.id, right.id]
        cluster_list += tree_to_hierarchical_cluster(left, left.count, cluster_set=cluster_list)
        cluster_list += tree_to_hierarchical_cluster(right, right.count, cluster_set=cluster_list)
    elif left.count > int(count*0.15):
        cluster_list += tree_to_hierarchical_cluster(left, count, cluster_set=cluster_list)
    elif right.count > int(count*0.15):
        cluster_list += tree_to_hierarchical_cluster(right, count, cluster_set=cluster_list)
    else:
        return cluster_list
    
    return cluster_list + [node.id]

from numpy.random import binomial
def tree_to_hierarchical_cluster(node, count, 
                                 cluster_set = set(), recursive_security=0):
    
    left, right = node.left, node.right
    if binomial(1,0.05):
        print(len(cluster_set))

    if node.count < 4:
        return cluster_set
    elif (type(left) == type(None)) or (type(right) == type(None)):
        return cluster_set
    
    
    
    if recursive_security > 30:
        print(recursive_security)
        print(len(cluster_set))
        sop += 1
    
    
    if ((min(left.count, right.count)/max(left.count,right.count)) > 0.1) and (min(left.count,right.count) > 1):
        cluster_set.union(set([left.id, right.id]))
        cluster_set.union(tree_to_hierarchical_cluster(left, left.count, cluster_set=cluster_set,recursive_security=recursive_security+1))
        cluster_set.union(tree_to_hierarchical_cluster(right, right.count, cluster_set=cluster_set,recursive_security=recursive_security+1))
    elif left.count > max(3,int(count*0.15)):
        cluster_set.union(tree_to_hierarchical_cluster(left, count, cluster_set=cluster_set,recursive_security=recursive_security+1))
    elif right.count > max(3,int(count*0.15)):
        cluster_set.union(tree_to_hierarchical_cluster(right, count, cluster_set=cluster_set,recursive_security=recursive_security+1))
    else:
        print(count*0.15)
        return cluster_set
    

    
    return cluster_set

from scipy.cluster.hierarchy import fclusterdata
def get_interesting_clusters(Z_in):
    clusters = []
    for i in range(2,Z.shape[0]-1):
        potential_clust = fclusterdata(Z, t=i,
                                  criterion="maxclust",
                                  metric="cosine")
        
        un, count = np.unique(potential_clust, return_counts=True)
        if sum(count > 4) > 3:
            clusters += [potential_clust]

    return clusters        
    
for pot in pot_clust:
    un, count = np.unique(pot,return_counts=True)

    entrop = lambda a : -a@np.log(a)

    print(entrop(count/count.sum()))