# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 18:11:31 2024

@author: hazem
"""


def create_sentence_node(list_sentences):
    nodes= list(range(0,len(list_sentences)))
    return nodes



#nodes and sentences of an article
def collect_equal_pairs(nodes,sentences):
    #counter to iterate over the upper triangle
    i=1
    #list of pairs
    lop=[]
    #checked nodes sent1=sent7  add sent7 then no need to check 7 equal with others
    #enode:equaled nodes
    enode=[]
    for node,sentence in zip(nodes,sentences):
        #ex: [1,7] when it comes to 7 no need to be checked already being equaled
        if node not in enode:
            for node2,sentence2 in zip(nodes[i:],sentences[i:]):
                if sentence==sentence2:
                    a=[node,node2]
                    enode.append(node2)
                    lop.append(a)
        i+=1
    return lop



#edges: list of sequential edges of an article
#LP: list of equality pairs [[1,6],[7,9]] means sentence1 =sentence6, sentence7=sentence9 
def updating_edges(LP,edges):
    #iterate over equality pairs
    for pair in LP:
        #iterate over edges
        for edge in edges:
            #replacing the source of edge of 2nd term with the 1st term
            if edge[0]==pair[1]:
                edge[0]=pair[0]
            #replacing the distination of edge of 2nd term with the 1st term   
            if edge[1]==pair[1]:
                edge[1]=pair[0]
    #remove duplication of the list
    updated_edges=list(set(tuple(edge) for edge in edges))
    #sorted by the source of the edge "might change it to x: (x[0],x[1])"
    updated_edges=sorted(updated_edges, key=lambda x: x[0])
    #updated_edges=[(x) for x in update_edges]
    return updated_edges

def delete_extra_nodes(nodes,article_EP):
        #get set of distination of Equaled pair (1,5)(7,9) ==>5,9
        dist_EP=[x[1] for x in article_EP]
        #[0,1,2...10]-[5,9] ==> [0..4,6,7,8,10]
        new_nodes=sorted(list(set(nodes)-set(dist_EP)))
        return new_nodes
    
def standrize_edges(edges):
    #get source nodes
    source=[x[0] for x in edges]
    #get distination nodes
    dist=[x[1] for x in edges]
    #get unique nodes
    nodes=sorted(list(set(source+dist)))
    #get new desired node
    newnodes=list(range(len(nodes)))
    #mapping the old node with desired node
    nodesmapping={x:y for x,y in zip(nodes,newnodes)}
    #print(nodesmapping)
    #change source nodes
    source=[nodesmapping[x] for x in source]
    #change distination nodes
    dist=[nodesmapping[x] for x in dist]
    #form edges with new nodes
    edges=[(a,b) for a,b in zip(source,dist)]
    
    return edges