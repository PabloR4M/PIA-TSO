from __future__ import annotations

import dataclasses
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Set
from collections import Counter
import networkx as nx

@dataclass(frozen=True)
class Edge:
    u: int
    v: int
    cost: int


@dataclass
class Instance:
    name: str
    comment: str
    vertex_count: int
    required_edges: List[Edge]
    non_required_edges: List[Edge]


def parse_instance_file(path: str) -> Instance:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    def read_header_value(prefix: str) -> str:
        for line in lines:
            if line.startswith(prefix):
                return line.split(":", 1)[1].strip()
        raise ValueError(f"Header {prefix} no encontrado en {path}")

    name = read_header_value("NOMBRE")
    comment = read_header_value("COMENTARIO")
    vertex_count = int(read_header_value("VERTICES"))

    required_edges: List[Edge] = []
    non_required_edges: List[Edge] = []

    edge_pattern = re.compile(
        r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s+coste\s+(\d+)"
    )

    def parse_edges(start_line: str) -> List[Edge]:
        edges: List[Edge] = []
        start_idx = None
        for idx, line in enumerate(lines):
            if line.startswith(start_line):
                start_idx = idx + 1
                break
        if start_idx is None:
            return edges

        for line in lines[start_idx:]:
            stripped = line.strip()
            if not stripped or not stripped.startswith("("):
                break
            match = edge_pattern.search(stripped)
            if not match:
                break
            u, v, cost = map(int, match.groups())
            edges.append(Edge(u, v, cost))
        return edges

    required_edges = parse_edges("LISTA_ARISTAS_REQ")
    non_required_edges = parse_edges("LISTA_ARISTAS_NOREQ")

    return Instance(
        name=name,
        comment=comment,
        vertex_count=vertex_count,
        required_edges=required_edges,
        non_required_edges=non_required_edges,
    )

def C_Heuristic(instance: Instance) -> int:
    # Returns: total cost of the solution
    # STEP 1: Build the full graph G and identify V+_R
    # V+_R = all nodes incident to required edges
    print("\n=== STEP 1: Building graph and computing shortest paths ===")
    
    # Build full graph G with all edges (required + non-required)
    G = nx.Graph()
    for edge in instance.required_edges:
        G.add_edge(edge.u, edge.v, weight=edge.cost)
    for edge in instance.non_required_edges:
        G.add_edge(edge.u, edge.v, weight=edge.cost)
    
    # Identify V+_R: all nodes that are endpoints of required edges
    V_plus_R: Set[int] = set()
    for edge in instance.required_edges:
        V_plus_R.add(edge.u)
        V_plus_R.add(edge.v)
    
    V_plus_R = sorted(V_plus_R)  # Sort for consistent ordering
    # print(f"V+_R has {len(V_plus_R)} nodes: {V_plus_R[:10]}..." if len(V_plus_R) > 10 else f"V+_R has {len(V_plus_R)} nodes: {V_plus_R}")
    
    # Compute shortest paths between all pairs in V+_R
    # Distance matrix: dist[i][j] = shortest path from V_plus_R[i] to V_plus_R[j]
    n = len(V_plus_R)
    dist_matrix: Dict[int, Dict[int, float]] = {}
    path_matrix: Dict[int, Dict[int, List[int]]] = {}  # Store actual paths
    
    print(f"Running Dijkstra from {n} nodes...")
    for i, source in enumerate(V_plus_R):
        # Run Dijkstra from source to all nodes
        distances, paths = nx.single_source_dijkstra(G, source, weight='weight')
        
        # Store distances and paths to all nodes in V+_R
        dist_matrix[source] = {}
        path_matrix[source] = {}
        for target in V_plus_R:
            dist_matrix[source][target] = distances.get(target, float('inf'))
            if target in paths:
                path_matrix[source][target] = paths[target]
        
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{n} Dijkstra runs...")
    
    print(f"✓ Shortest path matrix computed ({n}x{n})")
    
    # STEP 2: Build the reduced graph G'
    # G' is a complete graph on V+_R with edge weights = shortest path distances
    # Then remove redundant edges where c_ij = c_ik + c_jk for some k
    print("\n=== STEP 2: Building reduced graph G' ===")
    
    # Create complete graph on V+_R
    G_prime = nx.Graph()
    for i in V_plus_R:
        for j in V_plus_R:
            if i != j:
                cost = dist_matrix[i][j]
                if cost < float('inf'):
                    G_prime.add_edge(i, j, weight=cost)
    
    print(f"Initial G' has {G_prime.number_of_edges()} edges (complete graph on {n} nodes)")
    
    # Remove redundant edges: if c_ij = c_ik + c_jk for some k, delete edge (i,j)
    # This is the O(|V|³) bottleneck mentioned in the paper
    edges_to_remove = []
    print("Removing redundant edges (triangle inequality check)...")
    
    for i in V_plus_R:
        for j in V_plus_R:
            if i >= j or not G_prime.has_edge(i, j):
                continue
            
            c_ij = dist_matrix[i][j]
            
            # Check if there exists k such that c_ij = c_ik + c_jk
            for k in V_plus_R:
                if k == i or k == j:
                    continue
                
                c_ik = dist_matrix[i][k]
                c_jk = dist_matrix[j][k]
                
                # If triangle equality holds, edge (i,j) is redundant
                if c_ik < float('inf') and c_jk < float('inf'):
                    if abs(c_ij - (c_ik + c_jk)) < 1e-9:  # Floating point comparison
                        edges_to_remove.append((i, j))
                        break  # Found redundant, no need to check more k
    
    # Remove redundant edges
    for i, j in edges_to_remove:
        if G_prime.has_edge(i, j):
            G_prime.remove_edge(i, j)
    
    print(f"✓ Removed {len(edges_to_remove)} redundant edges")
    print(f"Final G' has {G_prime.number_of_edges()} edges")
    
    # STEP 3: Compute Minimum Spanning Tree (MST) on G'
    print("\n=== STEP 3: Computing MST on G' ===")
    
    # Compute MST - this connects all nodes in V+_R with minimum total cost
    mst_G_prime = nx.minimum_spanning_tree(G_prime, weight='weight')
    
    mst_edges = list(mst_G_prime.edges(data=True))
    mst_cost = sum(data['weight'] for _, _, data in mst_edges)
    
    print(f"✓ MST computed with {len(mst_edges)} edges")
    print(f"MST cost in G': {mst_cost:.2f}")
    
    # Store shortest paths for each MST edge (needed for Step 4)
    # We'll need to recover the actual paths in G for each MST edge
    mst_paths: Dict[tuple, List[int]] = {}
    
    print("Recovering actual paths for MST edges...")
    for u, v, data in mst_edges:
        # Get the path we already computed in Step 1
        if u in path_matrix and v in path_matrix[u]:
            path = path_matrix[u][v]
        else:
            # Fallback: compute if not found (shouldn't happen)
            path = nx.shortest_path(G, u, v, weight='weight')
        mst_paths[(u, v)] = path
    
    print(f"✓ All {len(mst_paths)} paths recovered")
    
    # STEP 4: Convert MST paths to edges in G (create deadheading set F)
    print("\n=== STEP 4: Creating deadheading edge set F ===")
    
    # F is a multiset of edges (can have duplicates if multiple paths share edges)
    F = Counter()
    
    for (u, v), path in mst_paths.items():
        # Convert path [u, n1, n2, ..., v] to edges: (u, n1), (n1, n2), ..., (n_{k-1}, v)
        for i in range(len(path) - 1):
            edge_u = path[i]
            edge_v = path[i + 1]
            # Store edge as tuple (min, max) for consistency
            edge = (min(edge_u, edge_v), max(edge_u, edge_v))
            F[edge] += 1
    
    print(f"✓ Created deadheading set F with {len(F)} unique edges")
    total_F_edges = sum(F.values())
    print(f"Total edge traversals in F: {total_F_edges}")
    
    # STEP 5: Add T-join to make the graph Eulerian
    print("\n=== STEP 5: Computing T-join to make graph Eulerian ===")
    
    # Build multigraph with required edges + deadheading edges
    M = nx.MultiGraph()
    M.add_nodes_from(G.nodes())
    
    # Add required edges (each once)
    for edge in instance.required_edges:
        M.add_edge(edge.u, edge.v)
    
    # Add deadheading edges (with multiplicities from F)
    for (u, v), count in F.items():
        for _ in range(count):
            M.add_edge(u, v)
    
    print(f"Multigraph M has {M.number_of_nodes()} nodes and {M.number_of_edges()} edges")
    
    # Find odd-degree nodes (T-set)
    odd_nodes = [v for v, d in M.degree() if d % 2 == 1]
    print(f"Found {len(odd_nodes)} odd-degree nodes")
    
    # If no odd-degree nodes, graph is already Eulerian
    if not odd_nodes:
        print("✓ Graph is already Eulerian (no odd-degree nodes)")
        # Compute total cost
        total_cost = 0.0
        for edge in instance.required_edges:
            total_cost += edge.cost
        for (u, v), count in F.items():
            if G.has_edge(u, v):
                total_cost += G[u][v]['weight'] * count
        print(f"\n=== SOLUTION ===")
        print(f"Total cost: {total_cost:.2f}")
        return int(total_cost)
    
    # Compute minimum T-join using minimum weight perfect matching
    # Build complete graph on odd-degree nodes with weights = shortest path distances
    print("Building complete graph on odd-degree nodes...")
    K = nx.Graph()
    for i, u in enumerate(odd_nodes):
        for j, v in enumerate(odd_nodes):
            if i < j:
                # Get shortest path distance
                if u in dist_matrix and v in dist_matrix[u]:
                    weight = dist_matrix[u][v]
                else:
                    # Fallback: compute if needed
                    weight = nx.shortest_path_length(G, u, v, weight='weight')
                K.add_edge(u, v, weight=weight)
    
    print(f"Complete graph K has {K.number_of_nodes()} nodes and {K.number_of_edges()} edges")
    
    # Find minimum weight perfect matching
    # NetworkX uses max_weight_matching, so we negate weights for minimum
    print("Computing minimum weight perfect matching...")
    K_neg = nx.Graph()
    for u, v, data in K.edges(data=True):
        K_neg.add_edge(u, v, weight=-data['weight'])
    
    matching_edges = nx.algorithms.matching.max_weight_matching(K_neg, weight='weight', maxcardinality=True)
    # Convert set of edges to list of tuples
    matching = list(matching_edges)
    
    print(f"✓ Matching computed with {len(matching)} pairs")
    
    # Add paths for each matched pair to make degrees even
    print("Adding T-join paths to multigraph...")
    for u, v in matching:
        # Get the shortest path
        if u in path_matrix and v in path_matrix[u]:
            path = path_matrix[u][v]
        else:
            path = nx.shortest_path(G, u, v, weight='weight')
        
        # Add all edges along the path
        for i in range(len(path) - 1):
            M.add_edge(path[i], path[i + 1])
    
    # Verify all degrees are now even
    odd_after = [v for v, d in M.degree() if d % 2 == 1]
    if odd_after:
        print(f"WARNING: Still have {len(odd_after)} odd-degree nodes after T-join!")
    else:
        print("✓ Graph is now Eulerian (all degrees are even)")
    
    # STEP 6: Compute total cost of the solution
    print("\n=== STEP 6: Computing total solution cost ===")
    
    total_cost = 0.0
    
    # Cost of required edges (each traversed once)
    for edge in instance.required_edges:
        total_cost += edge.cost
    
    # Cost of deadheading edges in F (with multiplicities)
    for (u, v), count in F.items():
        if G.has_edge(u, v):
            total_cost += G[u][v]['weight'] * count
    
    # Cost of T-join paths (if we added any)
    if odd_nodes:
        for u, v in matching:
            if u in dist_matrix and v in dist_matrix[u]:
                total_cost += dist_matrix[u][v]
            else:
                # Fallback
                total_cost += nx.shortest_path_length(G, u, v, weight='weight')
    
    print(f"\n=== SOLUTION ===")
    print(f"Total cost: {total_cost:.2f}")
    print(f"Required edges: {len(instance.required_edges)}")
    print(f"Deadheading edges (unique): {len(F)}")
    print(f"Deadheading edge traversals: {total_F_edges}")
    if odd_nodes:
        print(f"T-join pairs: {len(matching)}")
    
    return int(total_cost)

def Improved_C_Heuristic(instance: Instance) -> int:
    return 0 


def main() -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    inst_dir = os.path.join(project_root, "instancias")

    for idx in range(1, 11):
        name = f"GRP{idx}"
        file_path = os.path.join(inst_dir, name)
        if not os.path.isfile(file_path):
            print(f"No se encontró {file_path}")
            continue

        print("=" * 60)
        print(f"Instancia: {name}")
        instance = parse_instance_file(file_path)
        print(f"Required edges: {len(instance.required_edges)}")
        print(f"Non-required edges: {len(instance.non_required_edges)}")
        
        # Test C-heuristic (Step 1 only for now)
        cost = C_Heuristic(instance)
        print(f"\nSolution cost: {cost}")


if __name__ == "__main__":
    main()

