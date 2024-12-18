from pprint import pprint
import networkx

if __name__ == "__main__":
    number_of_vertices = 64

    G: networkx.Graph = networkx.barabasi_albert_graph(
        n=64, 
        m=12, 
        seed=42,
        initial_graph=None
    )
    # print(G.is_directed())

    human_readable_adj_list = [[] for idx in range(number_of_vertices)] 

    with open("/home/vladyslav/VSCodeProjects/briolette/src/simulation/briolettesim/graphs/barabasi_albert_test.txt", "w") as file:
        for edge in G.edges():
            human_readable_adj_list[edge[0]].append(edge[1])
            human_readable_adj_list[edge[1]].append(edge[0])
        
        for node_idx, node_neighbours in enumerate(human_readable_adj_list): 
            file.write(str(node_idx) + " ")
            file.write(' '.join([str(neighbour) for neighbour in node_neighbours]))
            file.write("\n")


    # networkx.write_adjlist(G, "barabasi_albert_test_compressed.txt")