from __future__ import annotations

import dataclasses
import os
import re
from dataclasses import dataclass
from typing import List

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
        for k, v in dataclasses.asdict(instance).items():
            print("-" * 40)
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()

