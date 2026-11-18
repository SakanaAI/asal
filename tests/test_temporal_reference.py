"""Reject stale simulator/scorer references without launching an evaluation."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.coralai_temporal import (ROOT, TemporalEvaluator, reference_contract,
                                     reference_files, reference_run, validate_reference)
from coralai import provenance


class TemporalReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / 'run'
        self.run.mkdir()
        self.config = json.loads((ROOT / 'configs/dynamic_energy_demo/dynamic.json').read_text())
        self.settings = json.loads((ROOT / 'configs/asal_novelty_pilot/search.json').read_text())
        self.contract = reference_contract(self.config, self.settings, self.settings['sampled_updates'], 'a' * 64)
        self.meta = {'effective_config': self.config, 'simulator_provenance': self.contract['simulator'],
                     'completed_steps': self.config['steps'], 'git_head': 'same-head'}
        self.captures = {}
        import hashlib
        for name in reference_files(self.settings['sampled_updates']):
            p = self.run / name
            p.parent.mkdir(exist_ok=True)
            p.write_bytes(name.encode())
            self.captures[name] = hashlib.sha256(p.read_bytes()).hexdigest()
        self.save()

    def save(self):
        (self.run / 'meta.json').write_text(json.dumps(self.meta))
        (self.run / 'reference_compatibility.json').write_text(json.dumps(
            {'contract': self.contract, 'capture_sha256': self.captures}))

    def reject(self, expected):
        with self.assertRaisesRegex(ValueError, 'Require fresh evaluation with --coralai_fresh_references'):
            validate_reference(self.run, expected)

    def test_matching_content_config_model_and_sampling_are_accepted(self):
        validate_reference(self.run, self.contract)
        changed_output = dict(self.config, output_dir='/another/destination')
        self.assertEqual(reference_contract(changed_output, self.settings, self.settings['sampled_updates'], 'a' * 64), self.contract)

    def test_matching_head_and_configuration_cannot_validate_legacy_reference(self):
        (self.run / 'reference_compatibility.json').write_text('{}')
        self.reject(self.contract)
        evaluator = TemporalEvaluator.__new__(TemporalEvaluator)
        evaluator.settings = self.settings
        evaluator.model_sha256 = 'a' * 64
        evaluator.simulate = Mock()
        destination = self.run.parent / 'evaluation'
        with self.assertRaisesRegex(ValueError, 'incompatible or unverifiable'):
            evaluator.evaluate(self.config, destination, reuse=self.run)
        self.assertFalse(destination.exists())
        evaluator.simulate.assert_not_called()

    def test_changed_uncommitted_simulator_content_is_rejected(self):
        expected = copy.deepcopy(self.contract)
        expected['simulator']['source_sha256']['coralai/evolver.py'] = '0' * 64
        self.reject(expected)  # Both records may still claim exactly the same Git HEAD.
        self.meta.pop('simulator_provenance')
        self.save()
        self.reject(self.contract)

    def test_all_effective_configuration_fields_are_checked(self):
        for key, value in [('seed', 999), ('infra_decay', 0.02), ('log_cell_events', True), ('screenshot_interval', 10)]:
            with self.subTest(key=key):
                expected = copy.deepcopy(self.contract)
                expected['effective_config'][key] = value
                self.reject(expected)

    def test_scoring_view_model_revision_weights_and_implementation_are_checked(self):
        for key, value in [('view', 'rgb'), ('model', 'another/model'), ('revision', 'b' * 40),
                           ('weights_sha256', 'b' * 64), ('implementation_sha256', {})]:
            with self.subTest(key=key):
                expected = copy.deepcopy(self.contract)
                expected['scoring'][key] = value
                self.reject(expected)

    def test_sampling_and_capture_integrity_are_checked(self):
        expected = copy.deepcopy(self.contract)
        expected['sampled_updates'] = [40, 200]
        self.reject(expected)
        next(iter(self.run.glob('frame_states/*.pt'))).write_bytes(b'changed')
        self.reject(self.contract)

    def test_fresh_reference_option_bypasses_an_existing_legacy_directory(self):
        demo = self.run.parent
        self.assertEqual(reference_run(demo, 'run'), self.run)
        self.assertIsNone(reference_run(demo, 'run', fresh=True))
        self.assertIsNone(reference_run(demo, 'missing'))

    def test_fingerprint_hashes_working_files_without_git_metadata(self):
        fake = self.run.parent / 'simulator'
        (fake / 'coralai').mkdir(parents=True)
        (fake / 'experiments/coral_dev').mkdir(parents=True)
        source = fake / 'coralai/evolver.py'
        source.write_text('controller = 1\n')
        (fake / 'experiments/coral_dev/neat.config').write_text('example\n')
        with patch.object(provenance, 'CORALAI_ROOT', fake), patch.dict(sys.modules, {'coralai.evolver': None}):
            first = provenance.simulator_fingerprint()
            source.write_text('controller = 2\n')
            second = provenance.simulator_fingerprint()
        self.assertNotEqual(first['source_sha256'], second['source_sha256'])
        self.assertEqual(first['package_versions'], second['package_versions'])


if __name__ == '__main__':
    unittest.main()
