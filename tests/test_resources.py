from types import SimpleNamespace
import unittest
from unittest.mock import patch

from movie_assistant.resources import GpuSampler, read_gpu, validate_worker_resources


class ResourceTests(unittest.TestCase):
    def test_gpu_units_parsed_and_unavailable_not_replaced_with_zero(self):
        with patch("movie_assistant.resources.shutil.which", return_value="nvidia-smi"), \
             patch("movie_assistant.resources.subprocess.run", return_value=SimpleNamespace(stdout="0, 42, 1024, 8192\n")):
            row = read_gpu()[0]
            self.assertEqual(row["memory_used_mib"], 1024)
            self.assertEqual(row["utilization_percent"], 42)
        with patch("movie_assistant.resources.shutil.which", return_value=None), self.assertRaises(OSError):
            read_gpu()
        self.assertIsNone(GpuSampler().report()["utilization_observed_peak_percent"])

    def test_sampler_reports_observed_peaks(self):
        sampler = GpuSampler()

        def reading():
            sampler.stop.set()
            return [{"gpu_index":0,"utilization_percent":42,"memory_used_mib":1024,"memory_total_mib":8192}]

        with patch("movie_assistant.resources.read_gpu", side_effect=reading):
            sampler.collect()
        report = sampler.report()
        self.assertEqual(report["utilization_observed_peak_percent"], 42)
        self.assertEqual(report["scope"], "whole_gpu_all_processes")
        self.assertEqual(len(report["samples"]), 1)

    def test_worker_measurements_reject_invalid_values(self):
        good = {"scope":"linux_worker_process", "peak_rss_bytes":1000,
                "cpu_user_seconds":.2, "cpu_system_seconds":.1}
        self.assertEqual(validate_worker_resources(good), good)
        for changed in ({"peak_rss_bytes":True}, {"scope":"whole_computer"},
                        {"cpu_user_seconds":float("nan")}, {"cpu_system_seconds":-1}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_worker_resources({**good, **changed})
