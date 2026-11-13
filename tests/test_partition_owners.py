"""Small, local-only topology/provenance checks; no data downloads."""

# Standard library
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Third-party
import torch

SPEC = importlib.util.spec_from_file_location(
    "partition_owners_tested",
    Path(__file__).resolve().parents[1] / "neural_lam/partition_owners.py",
)
PARTITION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARTITION)


class OwnerArtifactTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.mesh0 = torch.tensor([0, 0, 1, 1])
        self.write_graph()

    def write_graph(self, grid_first=False):
        g2m = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 0, 1, 2, 3, 3]])
        m2g = g2m.flip(0)
        if grid_first:
            g2m[1] += 6
            m2g[0] += 6
        else:
            g2m[0] += 4
            m2g[1] += 4
        m2m = [
            torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
            torch.tensor([[0, 1], [1, 0]]),
        ]
        up = [torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])]
        for name, value in {
            "mesh_features": [torch.zeros(4, 5), torch.ones(2, 5)],
            "m2m_edge_index": m2m,
            "g2m_edge_index": g2m,
            "m2g_edge_index": m2g,
            "mesh_up_edge_index": up,
            "mesh_down_edge_index": [up[0].flip(0)],
            "m2m_features": [torch.ones(4, 4), torch.ones(2, 4)],
            "g2m_features": torch.ones(6, 4),
            "m2g_features": torch.ones(6, 4),
            "mesh_up_features": [torch.ones(4, 4)],
            "mesh_down_features": [torch.ones(4, 4)],
        }.items():
            torch.save(value, self.root / f"{name}.pt")

    def test_mesh_first_and_grid_first_have_same_canonical_topology(self):
        counts, relations = PARTITION.read_topology(self.root)
        self.assertEqual(counts, {"mesh_0": 4, "mesh_1": 2, "grid": 6})
        self.write_graph(grid_first=True)
        other_counts, other = PARTITION.read_topology(self.root)
        self.assertEqual(counts, other_counts)
        self.assertEqual(set(relations), set(other))
        for key in relations:
            self.assertTrue(torch.equal(relations[key][2], other[key][2]))

    def test_extending_keeps_fine_owners_and_covers_all_relations(self):
        counts, relations = PARTITION.read_topology(self.root)
        owners = PARTITION.extend_owners(counts, relations, self.mesh0)
        self.assertTrue(torch.equal(owners["mesh_0"], self.mesh0))
        self.assertEqual(owners["grid"].tolist(), [0, 0, 0, 1, 1, 1])
        self.assertEqual(owners["mesh_1"].tolist(), [0, 1])
        result = PARTITION.relation_metrics(relations, owners)
        self.assertEqual(len(result), 6)
        self.assertEqual(result["g2m"]["directed_cut_edges"], 0)
        self.assertEqual(result["m2m_1"]["directed_cut_edges"], 2)
        for row in result.values():
            self.assertEqual(
                sum(r["receiver_edges"] for r in row["ranks"]),
                row["edge_count"],
            )

    def test_fingerprint_and_owner_validation_reject_stale_plans(self):
        counts, relations = PARTITION.read_topology(self.root)
        owners = PARTITION.extend_owners(counts, relations, self.mesh0)
        plan = {
            "schema_version": 2,
            "partition_count": 2,
            "graph_fingerprint": PARTITION.graph_fingerprint(self.root),
            "node_owner": owners,
        }
        path = self.root / "owners.pt"
        torch.save(plan, path)
        loaded = PARTITION.load_owners(path, self.root)
        self.assertTrue(torch.equal(loaded["grid"], owners["grid"]))
        torch.save(torch.zeros(6, 4), self.root / "g2m_features.pt")
        with self.assertRaisesRegex(ValueError, "fingerprints differ"):
            PARTITION.load_owners(path, self.root)

    def test_invalid_labels_and_namespaces_fail(self):
        counts, relations = PARTITION.read_topology(self.root)
        with self.assertRaises(ValueError):
            PARTITION.extend_owners(counts, relations, torch.tensor([0, 2]))
        plan = {
            "schema_version": 2,
            "partition_count": 2,
            "graph_fingerprint": PARTITION.graph_fingerprint(self.root),
            "node_owner": {"mesh_0": self.mesh0},
        }
        path = self.root / "owners.pt"
        torch.save(plan, path)
        with self.assertRaisesRegex(ValueError, "namespaces"):
            PARTITION.load_owners(path, self.root)

    def test_standalone_cli_round_trip_without_optional_partitioner(self):
        output = self.root / "generated"
        subprocess.run(
            [
                sys.executable,
                SPEC.origin,
                "--graph-dir",
                str(self.root),
                "--geometric",
                "--output-dir",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        owners = PARTITION.load_owners(output, self.root)
        self.assertEqual(owners["mesh_0"].tolist(), [0, 0, 1, 1])
        summary = json.loads((output / "owners.json").read_text())
        self.assertEqual(summary["mesh0_method"], "geometric_x_median")
        self.assertEqual(len(summary["relations"]), 6)

    def test_import_known_v2_rejects_another_graph_before_rebinding(self):
        counts, relations = PARTITION.read_topology(self.root)
        source = {
            "schema_version": 2,
            "partition_count": 2,
            "graph_fingerprint": PARTITION.graph_fingerprint(self.root),
            "node_owner": PARTITION.extend_owners(
                counts, relations, self.mesh0
            ),
        }
        path = self.root / "previous-owners.pt"
        torch.save(source, path)
        owner, provenance = PARTITION.load_mesh0_source(path, self.root, 4)
        self.assertTrue(torch.equal(owner, self.mesh0))
        self.assertEqual(
            provenance["mesh0_source_graph_verification"],
            "full_graph_fingerprint",
        )
        # The fine-mesh size stays unchanged; a grid-feature change alone
        # must not be laundered into a new full-graph fingerprint.
        torch.save(torch.zeros(6, 4), self.root / "g2m_features.pt")
        with self.assertRaisesRegex(ValueError, "fingerprints differ"):
            PARTITION.load_mesh0_source(path, self.root, 4)

    def test_import_legacy_export_checks_its_exact_two_file_hash(self):
        # This is the external v1 exporter contract: ordered basenames,
        # NUL separator and exact bytes of the two original graph files.
        digest = hashlib.sha256()
        for name in ("mesh_features.pt", "m2m_edge_index.pt"):
            digest.update(name.encode("utf-8") + b"\0")
            digest.update((self.root / name).read_bytes())
        source = {
            "schema_version": 1,
            "stage": "m2m",
            "mesh_level": 0,
            "node_types": ["mesh_0"],
            "edge_type": "m2m_0",
            "graph_hash": digest.hexdigest(),
            "node_owner": {"mesh_0": self.mesh0},
        }
        path = self.root / "legacy-metadata.pt"
        torch.save(source, path)
        owner, provenance = PARTITION.load_mesh0_source(path, self.root, 4)
        self.assertTrue(torch.equal(owner, self.mesh0))
        self.assertEqual(
            provenance["mesh0_source_graph_verification"],
            "legacy_mesh_geometry_and_m2m_only",
        )
        self.assertNotIn("mesh0_source_graph_fingerprint", provenance)
        # V1 never fingerprinted this file: its provenance must remain
        # explicitly partial rather than claiming whole-graph verification.
        torch.save(torch.zeros(6, 4), self.root / "g2m_features.pt")
        PARTITION.load_mesh0_source(path, self.root, 4)
        # An actual fine-mesh topology change, with the same node count,
        # is covered by v1 and must be rejected.
        torch.save(
            [torch.tensor([[0, 1], [2, 3]]), torch.tensor([[0, 1], [1, 0]])],
            self.root / "m2m_edge_index.pt",
        )
        with self.assertRaisesRegex(ValueError, "two-file SHA256"):
            PARTITION.load_mesh0_source(path, self.root, 4)

    def test_unknown_artifact_hash_format_requests_original_tensor(self):
        path = self.root / "unknown-owner-format.pt"
        torch.save(
            {
                "schema_version": 1,
                "graph_hash": "some-other-hash-format",
                "node_owner": {"mesh_0": self.mesh0},
            },
            path,
        )
        with self.assertRaisesRegex(ValueError, "original owner tensor"):
            PARTITION.load_mesh0_source(path, self.root, 4)

    def test_plain_tensor_cli_records_unverified_source_lineage(self):
        path = self.root / "mesh0_owner.pt"
        output = self.root / "imported"
        torch.save(self.mesh0, path)
        subprocess.run(
            [
                sys.executable,
                SPEC.origin,
                "--graph-dir",
                str(self.root),
                "--mesh0-owner",
                str(path),
                "--output-dir",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        summary = json.loads((output / "owners.json").read_text())
        self.assertEqual(
            summary["mesh0_source_graph_verification"],
            "unverified_plain_tensor",
        )
        self.assertNotIn("mesh0_source_graph_fingerprint", summary)
        self.assertNotIn("mesh0_source_legacy_graph_hash", summary)
        self.assertEqual(
            summary["graph_fingerprint"],
            PARTITION.graph_fingerprint(self.root),
        )
        self.assertTrue(
            torch.equal(
                PARTITION.load_owners(output, self.root)["mesh_0"],
                self.mesh0,
            )
        )


if __name__ == "__main__":
    unittest.main()
