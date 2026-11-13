"""Extend a two-way fine-mesh partition to the complete forecasting graph.

The additional owners are a deterministic connectivity-based baseline, not an
optimal partition. Original directed edges are preserved. The runtime builds
receiver-owned local/halo schedules from these typed owner vectors.
"""

# Standard library
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

# Third-party
import torch


def graph_fingerprint(graph_dir):
    """Fingerprint topology and static geometry, excluding generated plans."""
    root = Path(graph_dir)
    names = [
        "mesh_features.pt",
        "m2m_edge_index.pt",
        "g2m_edge_index.pt",
        "m2g_edge_index.pt",
        "m2m_features.pt",
        "g2m_features.pt",
        "m2g_features.pt",
        "mesh_up_edge_index.pt",
        "mesh_down_edge_index.pt",
        "mesh_up_features.pt",
        "mesh_down_features.pt",
    ]
    digest = hashlib.sha256()
    for name in names:
        path = root / name
        if not path.is_file():
            if name.startswith("mesh_up") or name.startswith("mesh_down"):
                continue
            raise ValueError(f"Missing graph file: {path}")
        digest.update(name.encode() + b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _edge_index(value, label):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{label}: expected an edge-index tensor")
    if value.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{label}: expected integer node IDs")
    if value.ndim != 2 or value.shape[0] != 2:
        raise ValueError(f"{label}: expected shape (2, E)")
    return value.to(device="cpu", dtype=torch.long)


def read_topology(graph_dir):
    """Read canonical node namespaces, matching the model's graph loader.

    This first integration supports graphs without additional auxiliary grid
    node groups. The runner checks owner sizes against the actual model.
    """
    root = Path(graph_dir)

    def load(name):
        return torch.load(root / name, map_location="cpu", weights_only=True)

    mesh = load("mesh_features.pt")
    if not isinstance(mesh, (list, tuple)) or not mesh:
        raise ValueError("Expected a nonempty list of mesh-level features")
    counts = {f"mesh_{i}": int(x.shape[0]) for i, x in enumerate(mesh)}
    m2m = load("m2m_edge_index.pt")
    if len(m2m) != len(mesh):
        raise ValueError("Different numbers of mesh-feature and edge levels")
    g2m = _edge_index(load("g2m_edge_index.pt"), "g2m")
    m2g = _edge_index(load("m2g_edge_index.pt"), "m2g")
    if not g2m.numel() or not m2g.numel():
        raise ValueError(
            "The initial graph bridge requires nonempty grid edges"
        )

    # Same mesh-first/grid-first convention as neural_lam.utils.load_graph.
    if int(m2g[0].min()) < int(m2g[1].min()):
        g2m = g2m.clone()
        m2g = m2g.clone()
        g2m[0] -= counts["mesh_0"]
        m2g[1] -= counts["mesh_0"]
    else:
        n_interior = int(m2g[1].max()) + 1
        n_grid = int(g2m[0].max()) + 1
        g2m = g2m.clone()
        m2g = m2g.clone()
        m2g[0] -= n_interior
        g2m[1] -= n_grid
    counts["grid"] = max(int(g2m[0].max()), int(m2g[1].max())) + 1
    relations = {"g2m": ("grid", "mesh_0", g2m), "m2g": ("mesh_0", "grid", m2g)}

    def add(name, source, target, value):
        index = _edge_index(value, name)
        # The existing model subtracts each row's minimum for these relations.
        # Match its effective topology, including its indexing convention.
        if index.numel():
            index = index - index.min(dim=1, keepdim=True).values
        relations[name] = (source, target, index)

    for i, index in enumerate(m2m):
        add(f"m2m_{i}", f"mesh_{i}", f"mesh_{i}", index)
    if len(mesh) > 1:
        up = load("mesh_up_edge_index.pt")
        down = load("mesh_down_edge_index.pt")
        if len(up) != len(mesh) - 1 or len(down) != len(up):
            raise ValueError("Wrong number of hierarchy transitions")
        for i, (up_index, down_index) in enumerate(zip(up, down)):
            add(f"up_{i}", f"mesh_{i}", f"mesh_{i + 1}", up_index)
            add(f"down_{i}", f"mesh_{i + 1}", f"mesh_{i}", down_index)
    for name, (source, target, index) in relations.items():
        if index.numel() and (
            int(index.min()) < 0
            or int(index[0].max()) >= counts[source]
            or int(index[1].max()) >= counts[target]
        ):
            raise ValueError(f"{name}: node IDs outside their namespace")
    return counts, relations


def _owners(value, count, label):
    if not isinstance(value, torch.Tensor) or value.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError(f"{label}: expected an integer owner tensor")
    value = value.to(device="cpu", dtype=torch.long)
    if (
        value.shape != (count,)
        or not torch.isin(value, torch.tensor([0, 1])).all()
    ):
        raise ValueError(f"{label}: expected {count} owners in {{0, 1}}")
    return value


def _choose_owners(votes):
    """Favor incident connections; resolve ties by current node load, then 0."""
    result = torch.empty(votes.shape[0], dtype=torch.long)
    loads = [0, 0]
    for node, (left, right) in enumerate(votes.tolist()):
        if left == right:
            rank = 0 if loads[0] <= loads[1] else 1
        else:
            rank = 0 if left > right else 1
        result[node] = rank
        loads[rank] += 1
    return result


def extend_owners(counts, relations, mesh0_owner):
    """Keep fine-mesh ownership and derive grid/coarser ownership by votes."""
    owners = {"mesh_0": _owners(mesh0_owner, counts["mesh_0"], "mesh_0")}
    votes = torch.zeros((counts["grid"], 2), dtype=torch.long)
    g2m = relations["g2m"][2]
    m2g = relations["m2g"][2]
    for grid_ids, mesh_ids in ((g2m[0], g2m[1]), (m2g[1], m2g[0])):
        flat = grid_ids * 2 + owners["mesh_0"][mesh_ids]
        votes += torch.bincount(flat, minlength=counts["grid"] * 2).reshape(
            -1, 2
        )
    owners["grid"] = _choose_owners(votes)
    for level in range(1, len(counts) - 1):
        fine, coarse = f"mesh_{level - 1}", f"mesh_{level}"
        index = relations[f"up_{level - 1}"][2]
        votes = torch.bincount(
            index[1] * 2 + owners[fine][index[0]],
            minlength=counts[coarse] * 2,
        ).reshape(-1, 2)
        owners[coarse] = _choose_owners(votes)
    return owners


def relation_metrics(relations, owners):
    """Report separate work/communication counts for every directed relation."""
    result = {}
    for name, (source, target, index) in relations.items():
        src, dst = index
        source_owner, target_owner = owners[source], owners[target]
        crossing = source_owner[src] != target_owner[dst]
        result[name] = {
            "source_type": source,
            "target_type": target,
            "edge_count": int(src.numel()),
            "directed_cut_edges": int(crossing.sum()),
            "ranks": [
                {
                    "rank": rank,
                    "receiver_edges": int((target_owner[dst] == rank).sum()),
                    "halo_nodes": int(
                        torch.unique(
                            src[crossing & (target_owner[dst] == rank)]
                        ).numel()
                    ),
                }
                for rank in (0, 1)
            ],
        }
    return result


def metis_mesh0(counts, relations, seed, balance="nodes"):
    """Build the initial fine-mesh owners inside the implementation project."""
    try:
        # Third-party
        import pymetis
    except ImportError as error:
        raise RuntimeError(
            'Install the optional partitioner: pip install "pymetis==2025.2.2"'
        ) from error
    size = counts["mesh_0"]
    adjacency = [set() for _ in range(size)]
    for sender, receiver in relations["m2m_0"][2].t().tolist():
        if sender != receiver:
            adjacency[sender].add(receiver)
            adjacency[receiver].add(sender)
    weights = None
    if balance == "g2m":
        weights = (
            torch.bincount(relations["g2m"][2][1], minlength=size)
            .clamp_min(1)
            .tolist()
        )
    _, membership = pymetis.part_graph(
        2,
        adjacency=[sorted(row) for row in adjacency],
        vweights=weights,
        recursive=True,
        options=pymetis.Options(seed=seed, ufactor=1),
    )
    return torch.tensor(membership, dtype=torch.long)


def geometric_mesh0(graph_dir):
    """Deterministic spherical-x median; needs no optional partitioner."""
    features = torch.load(
        Path(graph_dir) / "mesh_features.pt",
        map_location="cpu",
        weights_only=True,
    )[0]
    if features.ndim != 2 or features.shape[1] != 5:
        raise ValueError("Geometric baseline requires five static features")
    # Saved columns: sin(lon), cos(lon), sin(lat), cos(lat), area.
    order = torch.argsort(features[:, 3] * features[:, 1], stable=True)
    owners = torch.zeros(features.shape[0], dtype=torch.long)
    owners[order[features.shape[0] // 2 :]] = 1
    return owners


def load_owners(path, graph_dir_path):
    """Load a whole-graph plan and reject a different topology/geometry."""
    path = Path(path)
    if path.is_dir():
        path = path / "owners.pt"
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("schema_version") != 2 or data.get("partition_count") != 2:
        raise ValueError("Expected a version-2, two-rank owner artifact")
    if data.get("graph_fingerprint") != graph_fingerprint(graph_dir_path):
        raise ValueError("Owner plan and model graph fingerprints differ")
    counts, _ = read_topology(graph_dir_path)
    owners = data["node_owner"]
    if set(owners) != set(counts):
        raise ValueError("Owner plan has wrong node-type namespaces")
    return {name: _owners(owners[name], n, name) for name, n in counts.items()}


def _legacy_mesh0_fingerprint(graph_dir):
    """Match export_partition_metadata.py's original two-file graph_hash.

    Version 1 covered mesh geometry and m2m connectivity only. In particular,
    it cannot establish the identity of grid or hierarchy-transfer files.
    """
    digest = hashlib.sha256()
    for name in ("mesh_features.pt", "m2m_edge_index.pt"):
        path = Path(graph_dir) / name
        digest.update(path.name.encode("utf-8") + b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def load_mesh0_source(path, graph_dir, count):
    """Import fine owners without discarding a known graph-identity guard."""
    path = Path(path)
    data = torch.load(path, map_location="cpu", weights_only=True)
    provenance = {
        "mesh0_method": "imported_owner_vector",
        "mesh0_owner_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if isinstance(data, torch.Tensor):
        source = data
        provenance["mesh0_source_graph_verification"] = (
            "unverified_plain_tensor"
        )
    elif isinstance(data, dict) and data.get("schema_version") == 2:
        # Reuse the whole-artifact validator, including type/count checks.
        source = load_owners(path, graph_dir)["mesh_0"]
        provenance.update(
            mesh0_source_graph_verification="full_graph_fingerprint",
            mesh0_source_graph_fingerprint=data["graph_fingerprint"],
            mesh0_source_schema_version=2,
        )
    elif isinstance(data, dict) and (
        data.get("schema_version") == 1
        and data.get("stage") == "m2m"
        and data.get("mesh_level") == 0
        and data.get("node_types") == ["mesh_0"]
        and data.get("edge_type") == "m2m_0"
    ):
        supplied = data.get("graph_hash")
        if supplied != _legacy_mesh0_fingerprint(graph_dir):
            raise ValueError(
                "Version-1 graph_hash does not match the original two-file "
                "SHA256 format or this graph. Supply a matching artifact; "
                "for an unknown hash format, supply the original owner "
                "tensor, whose graph lineage will be marked unverified."
            )
        source = data.get("node_owner", {}).get("mesh_0")
        provenance.update(
            mesh0_source_graph_verification=(
                "legacy_mesh_geometry_and_m2m_only"
            ),
            mesh0_source_legacy_graph_hash=supplied,
            mesh0_source_schema_version=1,
        )
    else:
        raise ValueError(
            "Unsupported fine-mesh owner artifact or unknown version-1 "
            "hash format. Supply the original owner tensor, whose graph "
            "lineage will be marked unverified."
        )
    return _owners(source, count, "mesh_0"), provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    initial = parser.add_mutually_exclusive_group()
    initial.add_argument("--mesh0-owner", type=Path)
    initial.add_argument("--metis-seed", type=int)
    initial.add_argument("--geometric", action="store_true")
    parser.add_argument(
        "--mesh0-balance", choices=("nodes", "g2m"), default="nodes"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    counts, relations = read_topology(args.graph_dir)
    if args.mesh0_owner is not None:
        source, provenance = load_mesh0_source(
            args.mesh0_owner, args.graph_dir, counts["mesh_0"]
        )
    elif args.geometric:
        source = geometric_mesh0(args.graph_dir)
        provenance = {"mesh0_method": "geometric_x_median"}
    else:
        seed = 9 if args.metis_seed is None else args.metis_seed
        source = metis_mesh0(counts, relations, seed, args.mesh0_balance)
        provenance = {
            "mesh0_method": "metis_recursive",
            "partition_seed": seed,
            "mesh0_balance": args.mesh0_balance,
            "pymetis_version": version("pymetis"),
        }
    if set(source.tolist()) != {0, 1}:
        raise ValueError("The fine mesh must use both ranks")
    revision_path = args.graph_dir.parent.parent / "dataset-revision.txt"
    revision = (
        revision_path.read_text().strip() if revision_path.exists() else None
    )
    source_metrics = (
        args.mesh0_owner.parent / "metrics.json"
        if args.mesh0_owner is not None
        else None
    )
    if source_metrics is not None and source_metrics.exists():
        measured = json.loads(source_metrics.read_text()).get(
            "dataset_revision"
        )
        if (
            measured is not None
            and revision is not None
            and measured != revision
        ):
            raise ValueError("Fine-mesh partition and graph revisions differ")
    owners = extend_owners(counts, relations, source)
    data = {
        "schema_version": 2,
        "partition_count": 2,
        "method": "fixed_mesh0_incident_edge_votes",
        "scope": "global_boundary_bridge",
        "graph_fingerprint": graph_fingerprint(args.graph_dir),
        "dataset_revision": revision,
        **provenance,
        "node_owner": owners,
    }
    summary = {key: value for key, value in data.items() if key != "node_owner"}
    summary["nodes"] = {
        name: [int((owner == rank).sum()) for rank in (0, 1)]
        for name, owner in owners.items()
    }
    summary["relations"] = relation_metrics(relations, owners)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "owners.pt"
    torch.save(data, target)
    restored = load_owners(target, args.graph_dir)
    if any(not torch.equal(restored[k], v) for k, v in owners.items()):
        raise ValueError("Owner round trip failed")
    (args.output_dir / "owners.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved and validated: {target}")


if __name__ == "__main__":
    main()
