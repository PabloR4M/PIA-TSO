import sys
import networkx as nx
from collections import defaultdict, Counter


# ===========================================================
# === FASE 0: LECTOR DE INSTANCIAS ==========================
# Lee nodos, aristas, nodos requeridos (VR) y aristas requeridas (ER)
# Formato unificado para casos del GRP, STSP, RPP, etc.
# ===========================================================
def read_instance(path):
    with open(path) as f:
        header = f.readline().strip()

        # Saltar comentarios o líneas vacías
        while header.startswith('#') or not header:
            header = f.readline().strip()

        # n = nodos | m = aristas | r = cantidad de elementos requeridos
        n, m, r = map(int, header.split())

        # Construimos el grafo principal G
        G = nx.Graph()
        for _ in range(m):
            u, v, c = f.readline().split()
            G.add_edge(int(u), int(v), weight=float(c))

        # VR = nodos requeridos, ER = aristas requeridas
        VR = set()
        ER = set()

        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Los nodos requeridos se listan así: VR: 1 4 8 ...
            if line.startswith('VR:'):
                VR.update(map(int, line[3:].strip().split()))

            # Las aristas requeridas vienen como pares "u v"
            else:
                a, b = map(int, line.split())
                ER.add((a, b))

        # Devuelve grafo + requerimientos
        return G, VR, ER



# ===========================================================
# === FASE 1: COMPONENTES REQUERIDAS ========================
# Agrupa nodos/aristas requeridas en "supernodos" o componentes.
# Esta fase reduce el problema antes de aplicar el MST.
# ===========================================================
def compute_R_components(G, VR, ER):

    # Conjunto de nodos relevantes: nodos requeridos + nodos incidentes a aristas requeridas
    Vp = set(VR)
    for (u, v) in ER:
        Vp.add(u)
        Vp.add(v)

    # Grafo auxiliar que contiene sólo lo requerido
    H = nx.Graph()
    H.add_nodes_from(Vp)
    for (u, v) in ER:
        H.add_edge(u, v)

    # Obtener componentes conectadas de elementos requeridos
    components = list(nx.connected_components(H)) if H.number_of_edges() > 0 else [{v} for v in VR]

    # Diccionario: nodo → id de componente
    comp_of = {}
    for i, C in enumerate(components):
        for v in C:
            comp_of[v] = i

    # Aristas requeridas agrupadas por componente
    E_of = defaultdict(set)
    for (u, v) in ER:
        cid = comp_of[u]
        E_of[cid].add((u, v))

    return components, comp_of, E_of



# ===========================================================
# === FASE 2: CAMINOS MÍNIMOS ENTRE COMPONENTES =============
# Realiza la fase central del heurístico mejorado:
# Para cada componente requerida, corre Dijkstra y obtiene
# distancias mínimas hacia todas las demás.
# ===========================================================
def shortest_path_phase(G, components, comp_of, E_of):

    k = len(components)
    c = [[float("inf")] * k for _ in range(k)]  # matriz de distancias entre componentes
    T_trees = {}  # guarda distancias y caminos de Dijkstra por componente

    for i in range(k):
        Ci = components[i]
        v = next(iter(Ci))               # nodo representativo
        H = G.copy()

        # Las aristas requeridas dentro de la componente tienen costo 0
        # (ya son obligatorias)
        for (a, b) in E_of.get(i, []):
            if H.has_edge(a, b):
                H[a][b]["weight"] = 0.0

        # Dijkstra desde un nodo representativo
        dist, paths = nx.single_source_dijkstra(H, v, weight="weight")
        T_trees[i] = (dist, paths)

        # Calcular costos mínimos hacia cada componente j
        for j in range(k):
            if j == i:
                c[i][j] = 0.0
                continue

            # Escoger el nodo más accesible de la componente j
            best = float("inf")
            for node in components[j]:
                d = dist.get(node, float("inf"))
                if d < best:
                    best = d
            c[i][j] = best

    return c, T_trees



# ===========================================================
# === FASE 3: GRAFO REDUCIDO ENTRE COMPONENTES ==============
# Construye el grafo "shrunken" donde cada componente es un nodo.
# ===========================================================
def build_shrunk_graph(c):
    k = len(c)
    H = nx.Graph()
    H.add_nodes_from(range(k))

    # Se agregan aristas entre componentes con peso = distancia mínima
    for i in range(k):
        for j in range(i + 1, k):
            if c[i][j] < float("inf"):
                H.add_edge(i, j, weight=c[i][j])

    return H



# ===========================================================
# === FASE 4: MST SOBRE EL GRAFO REDUCIDO ===================
# Después de construir el grafo reducido, sacamos el MST.
# Luego este MST debe traducirse al grafo real.
# ===========================================================
def map_tree_to_G(mst, T_trees, components):

    Fprime = []  # F′ = conjunto preliminar de aristas reales

    # Para cada arista del MST entre componentes (i, j)
    for (i, j, data) in mst.edges(data=True):

        # Siempre se usa el árbol de Dijkstra almacenado en T_trees
        if i in T_trees:
            dist, paths = T_trees[i]

            # Buscar nodo real más cercano perteneciente a la componente j
            best_node = None
            best_d = float("inf")
            for node in components[j]:
                d = dist.get(node, float("inf"))
                if d < best_d:
                    best_d = d
                    best_node = node

            # Recuperar el camino real desde el nodo representativo de i
            path_nodes = paths.get(best_node)
            if path_nodes:
                for a, b in zip(path_nodes, path_nodes[1:]):
                    Fprime.append((a, b))

        else:
            # Lo mismo, pero al revés
            dist, paths = T_trees[j]
            best_node = None
            best_d = float("inf")
            for node in components[i]:
                d = dist.get(node, float("inf"))
                if d < best_d:
                    best_d = d
                    best_node = node

            path_nodes = paths.get(best_node)
            if path_nodes:
                for a, b in zip(path_nodes, path_nodes[1:]):
                    Fprime.append((a, b))

    return Counter(Fprime)   # multiset de aristas (cuenta duplicadas)



# ===========================================================
# === FASE 5: SPARSIFICACIÓN DE F′ ==========================
# Aquí se reduce el conjunto preliminar usando un MST
# para evitar duplicación innecesaria de aristas.
# ===========================================================
def sparsify_and_get_Fminus(G, VR, ER, components, comp_of, Fprime_counter):

    K = len(components)
    H = nx.Graph()
    H.add_nodes_from(range(K))

    # Función auxiliar que agrega la "mejor" arista entre componentes
    def add_edge_between(u, v, weight):

        i = comp_of.get(u)
        j = comp_of.get(v)
        if i is None or j is None or i == j:
            return

        # Si ya existe, quedarse con la más barata
        if H.has_edge(i, j):
            if weight < H[i][j]["weight"]:
                H[i][j]["weight"] = weight
                H[i][j]["rep"] = (u, v)
        else:
            H.add_edge(i, j, weight=weight, rep=(u, v))

    # Insertar aristas requeridas
    for (u, v) in ER:
        w = G[u][v]["weight"] if G.has_edge(u, v) else float("inf")
        add_edge_between(u, v, w)

    # Insertar aristas obtenidas desde F′
    for (u, v), count in Fprime_counter.items():
        if G.has_edge(u, v):
            w = G[u][v]["weight"]
            add_edge_between(u, v, w)

    # MST del grafo reducido refinado = F-
    mst = nx.minimum_spanning_tree(H, weight="weight")
    Fminus = set()

    for (i, j, data) in mst.edges(data=True):
        rep = data.get("rep")
        if rep:
            Fminus.add(rep)  # arista real necesaria

    return Fminus



# ===========================================================
# === FASE 6 y 7: T-JOIN PARA PARIFICAR Y FINALIZAR =========
# Construimos un multigrafo con ER + F- y luego añadimos
# un T-Join mínimo para lograr que todos los grados sean pares.
# ===========================================================
def add_T_join_and_finalize(G, ER, Fminus):

    # Multigrafo con aristas requeridas + sparsificadas
    M = nx.MultiGraph()
    M.add_nodes_from(G.nodes())

    for (u, v) in ER:
        M.add_edge(u, v)

    for (u, v) in Fminus:
        M.add_edge(u, v)

    # Detectar nodos de grado impar (T-set)
    odd = [v for v, d in M.degree() if d % 2 == 1]

    # Si no hay impares → ya es euleriano
    if not odd:
        return list(M.edges()), sum(G[u][v]["weight"] for u, v in M.edges())

    # Calcular distancias entre nodos impares
    dist = {}
    for s in odd:
        d, _ = nx.single_source_dijkstra(G, s, weight="weight")
        dist[s] = d

    # Grafo completo entre nodos impares con pesos = distancias min
    K = nx.Graph()
    for i in range(len(odd)):
        for j in range(i + 1, len(odd)):
            u = odd[i]
            v = odd[j]
            w = dist[u].get(v, float("inf"))
            K.add_edge(u, v, weight=w)

    # Matching mínimo = T-Join de costo mínimo
    mate = nx.algorithms.matching.min_weight_matching(
        K, maxcardinality=True, weight="weight")

    # Añadir caminos reales entre pares emparejados
    for (u, v) in mate:
        path = nx.shortest_path(G, u, v, weight="weight")
        for a, b in zip(path, path[1:]):
            M.add_edge(a, b)

    # Calcular costo total
    total_cost = 0.0
    for u, v in M.edges():
        total_cost += G[u][v]["weight"]

    return list(M.edges()), total_cost



# ===========================================================
# === FASE 8–10: ORQUESTACIÓN DEL ALGORITMO =================
# Ejecuta todas las fases del heurístico mejorado
# en el orden del artículo
# ===========================================================
def improved_c_heuristic_instance(G, VR, ER):

    # 1. Componentes requeridas
    components, comp_of, E_of = compute_R_components(G, VR, ER)

    # 2. Caminos mínimos entre componentes
    c, T_trees = shortest_path_phase(G, components, comp_of, E_of)

    # 3. Grafo reducido
    shrunk = build_shrunk_graph(c)

    # 4. MST del grafo reducido
    mst = nx.minimum_spanning_tree(shrunk, weight="weight")

    # 5. Mapear MST al grafo real → F′
    Fprime = map_tree_to_G(mst, T_trees, components)

    # 6. Sparsificar F′ → F-
    Fminus = sparsify_and_get_Fminus(G, VR, ER, components, comp_of, Fprime)

    # 7. T-Join + Eulerian final
    final_edges, cost = add_T_join_and_finalize(G, ER, Fminus)

    return final_edges, cost



# ===========================================================
# === MAIN ==================================================
# Llamada principal desde terminal.
# ===========================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python solve_grp.py instancia.txt")
        sys.exit(0)

    file_path = sys.argv[1]

    # Leer instancia del archivo
    G, VR, ER = read_instance(file_path)

    print(f"Instancia cargada. Nodos: {len(G.nodes())}, Aristas: {len(G.edges())}")
    print(f"Nodos requeridos: {VR if VR else 'Ninguno'}")
    print(f"Aristas requeridas: {len(ER)}")

    # Ejecutar heurístico completo
    edges, cost = improved_c_heuristic_instance(G, VR, ER)

    print(f"\nCosto total estimado: {cost:.2f}")
    print("Cantidad de aristas en la solución:", len(edges))
