"""Schedules built by the original pack's copy of this module still resolve.

This node is meant to be installed beside cicalooo/ComfyUI-H3-PowerLoraStack and
driven by that pack's **MiniMax H3 LoRA Schedule**.  Its ``Schedule`` is the
same frozen dataclass, but ComfyUI imports each pack under its own top-level
name, so the two classes are not the same object and ``isinstance`` says no.
The chain is loaded here the way ComfyUI would load the other pack -- a second,
independent module object from the same source file -- rather than faked, so
the test would notice if ``resolve`` went back to an identity check.
"""

import importlib.util
import sys
import unittest

from h3lora import schedule

FOREIGN = "h3lora_from_the_other_pack.schedule"


def load_foreign_schedule_module():
    """``h3lora.schedule`` again, as a module the ``from h3lora`` one never met."""
    spec = importlib.util.spec_from_file_location(FOREIGN, schedule.__file__)
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations through sys.modules[cls.__module__], so
    # the module has to be registered before its body runs -- exactly what a
    # normal import does.
    sys.modules[FOREIGN] = module
    spec.loader.exec_module(module)
    return module


class ForeignScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.other = load_foreign_schedule_module()

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop(FOREIGN, None)

    def test_the_two_schedule_classes_really_are_distinct(self):
        self.assertIsNot(self.other.Schedule, schedule.Schedule)
        self.assertNotIsInstance(self.other.Schedule(), schedule.Schedule)

    def test_a_foreign_chain_selects_rows(self):
        chain = (self.other.Schedule(rows="2-3", start_strength=1.0,
                                     end_strength=0.0),)
        self.assertIsNone(schedule.resolve(chain, 1))
        self.assertIs(schedule.resolve(chain, 2), chain[0])
        self.assertIs(schedule.resolve(chain, 3), chain[0])
        self.assertIsNone(schedule.resolve(chain, 4))

    def test_the_later_foreign_link_wins_on_an_overlap(self):
        first = self.other.Schedule(rows="all")
        second = self.other.Schedule(rows="1")
        self.assertIs(schedule.resolve((first, second), 1), second)
        self.assertIs(schedule.resolve((first, second), 2), first)

    def test_a_lone_foreign_schedule_is_treated_as_a_one_link_chain(self):
        lone = self.other.Schedule(rows="all")
        self.assertIs(schedule.resolve(lone, 1), lone)

    def test_a_foreign_link_still_evaluates(self):
        link = self.other.Schedule(rows="all", start_strength=1.0,
                                   end_strength=0.0, curve="linear")
        resolved = schedule.resolve((link,), 1)
        self.assertAlmostEqual(resolved.evaluate(0.0), 1.0)
        self.assertAlmostEqual(resolved.evaluate(1.0), 0.0)
        self.assertEqual(resolved.domain, "steps")

    def test_junk_on_the_socket_is_ignored_rather_than_crashing(self):
        self.assertIsNone(schedule.resolve(None, 1))
        self.assertIsNone(schedule.resolve((), 1))
        self.assertIsNone(schedule.resolve(("not a schedule", 7), 1))


if __name__ == "__main__":
    unittest.main()
