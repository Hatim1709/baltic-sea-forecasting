"""CPU recovery tests with fake training and no dataset dependencies."""

import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))
import run_ckpt_policy_benchmark as benchmark
import resume_ckpt_policy_benchmark as resume

FAKE_TRAIN = textwrap.dedent('''
import csv
from pathlib import Path
import sys
import time
def arg(name):
    return sys.argv[sys.argv.index(name) + 1]
policy = arg('--ckpt_policy')
with open('started.txt', 'a') as handle:
    handle.write(policy + '\\n')
factor = {'off': 1, 'unroll': 2, 'internal': 2, 'unroll+internal': 3}[policy]
blocks = {'off': 0, 'unroll': 0, 'internal': 16, 'unroll+internal': 32}[policy]
name = f'step-measurements-{time.time_ns()}.csv'
with open(name, 'w', newline='') as handle:
    writer = csv.writer(handle)
    writer.writerow(['step','phase','epoch','step_ms','allocated_mib','reserved_mib',
                     'checkpointing','ckpt_policy','internal_ckpt','layer_calls',
                     'layer_blocks','unroll_blocks','loss'])
    for i in range(1, int(arg('--limit_train_batches')) + 1):
        writer.writerow([i,'warmup' if i <= 5 else 'measured',0,100*factor,1000,1200,
                         policy.startswith('unroll'),policy,policy.endswith('internal'),
                         16*factor,blocks,int(arg('--ar_steps_train')) if policy.startswith('unroll') else 0,1.0])
print('Measurements CSV:', name)
print('`Trainer.fit` stopped: `max_epochs=5` reached.')
''')


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.source = root / "source"
        self.work = root / "work"
        self.session = root / "session"
        for folder in [self.source, self.work, self.session]:
            folder.mkdir()
        (self.source / "fake_train.py").write_text(FAKE_TRAIN)
        self.args = benchmark.parse_args([
            "--source", str(self.source), "--workdir", str(self.work),
            "--launches", "2", "--measured", "2", "--train_module", "fake_train",
        ])
        self.plan = benchmark.build_plan(self.args.policies, self.args.launches, self.args.seed)
        manifest = {"args": vars(self.args), "plan": self.plan, "provenance": {}}
        (self.session / "manifest.json").write_text(json.dumps(manifest))
        self.source_check = patch.object(resume, "check_model_source")
        self.source_check.start()
        self.addCleanup(self.source_check.stop)
        self.bootstrap = patch.object(benchmark, "BOOTSTRAP_DRAWS", 20)
        self.bootstrap.start()
        self.addCleanup(self.bootstrap.stop)
        benchmark.run_launch(self.plan[0], self.args, self.session)

    def run_resume(self, dry_run=False):
        with contextlib.redirect_stdout(io.StringIO()):
            resume.continue_session(self.session, dry_run)

    def test_recover_then_continue_without_repeating_first_launch(self):
        self.run_resume(dry_run=True)
        self.assertEqual(len((self.work / "started.txt").read_text().splitlines()), 1)
        records = json.loads((self.session / "resume_records.json").read_text())
        self.assertIsNone(records[0]["returncode"])
        self.assertIsNone(records[0]["process_seconds"])
        self.assertTrue((self.session / self.plan[0]["run"] / "recovery.json").exists())
        self.run_resume()
        self.run_resume()
        self.assertEqual(len((self.work / "started.txt").read_text().splitlines()), 8)
        rows = benchmark.read_csv(self.session / "launches.csv")
        self.assertEqual([row["run"] for row in rows], [item["run"] for item in self.plan])
        self.assertTrue(all(row["status"] == "completed" for row in rows))

    def test_incomplete_unjournaled_launch_is_preserved_and_refused(self):
        log = self.session / self.plan[0]["run"] / "train.log"
        log.write_text(log.read_text().replace("`Trainer.fit` stopped:", "interrupted"))
        with self.assertRaisesRegex(RuntimeError, "completion log"):
            self.run_resume()
        self.assertEqual(len((self.work / "started.txt").read_text().splitlines()), 1)

    def test_invalid_policy_counters_are_refused(self):
        csv_file, = (self.session / self.plan[0]["run"]).glob("step-measurements-*.csv")
        csv_file.write_text(csv_file.read_text().replace(",48,32,2,", ",16,0,0,"))
        with self.assertRaisesRegex(RuntimeError, "failed audit"):
            self.run_resume(dry_run=True)

    def test_resume_after_additional_launches_have_been_journaled(self):
        original = benchmark.run_launch
        def interrupted(item, args, session):
            if item == self.plan[3]:
                raise RuntimeError("Simulated interruption before next launch")
            return original(item, args, session)
        with patch.object(benchmark, "run_launch", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "Simulated interruption"):
                self.run_resume()
        self.run_resume()
        self.assertEqual(len((self.work / "started.txt").read_text().splitlines()), 8)


if __name__ == "__main__":
    unittest.main()
