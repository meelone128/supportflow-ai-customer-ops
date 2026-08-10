import unittest
from threading import Event

from supportflow.dispatch import InlineTicketDispatcher, LocalTicketDispatcher


class RecordingExecutor:
    def __init__(self):
        self.called = Event()
        self.arguments = None

    def execute(self, ticket_id: str, recovered_from_restart: bool = False):
        self.arguments = (ticket_id, recovered_from_restart)
        self.called.set()


class DispatchTests(unittest.TestCase):
    def test_local_dispatcher_runs_the_shared_executor_in_the_background(self):
        executor = RecordingExecutor()
        LocalTicketDispatcher(executor).dispatch("T-0042", recovered_from_restart=True)
        self.assertTrue(executor.called.wait(timeout=1))
        self.assertEqual(executor.arguments, ("T-0042", True))

    def test_inline_dispatcher_runs_the_shared_executor_before_returning(self):
        executor = RecordingExecutor()
        InlineTicketDispatcher(executor).dispatch("T-0043", recovered_from_restart=False)
        self.assertTrue(executor.called.is_set())
        self.assertEqual(executor.arguments, ("T-0043", False))
