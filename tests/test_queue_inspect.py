import numpy as np
import pytest

from rfdes import Component, HeapScheduler, RFSystem
from rfdes.components import Amplifier, Recorder, Splitter
from rfdes.scheduler import labeled


# -- scheduler-level introspection -------------------------------------------
def test_pending_lists_events_in_fire_order_with_labels():
    sched = HeapScheduler()
    sched.schedule(3.0, lambda: None, label="c")
    sched.schedule(1.0, lambda: None, label="a")
    sched.schedule(2.0, lambda: None, label="b")
    assert sched.pending() == [(1.0, "a"), (2.0, "b"), (3.0, "c")]


def test_pending_excludes_cancelled():
    sched = HeapScheduler()
    sched.schedule(1.0, lambda: None, label="keep")
    h = sched.schedule(2.0, lambda: None, label="drop")
    sched.cancel(h)
    assert sched.pending() == [(1.0, "keep")]


def test_pending_reads_label_from_callback_attribute():
    sched = HeapScheduler()
    cb = labeled(lambda: None, "tagged")
    sched.schedule(5.0, cb)  # no label kwarg; picked up from the attribute
    assert sched.pending() == [(5.0, "tagged")]


def test_unlabeled_events_show_generic_label():
    sched = HeapScheduler()
    sched.schedule(1.0, lambda: None)
    assert sched.pending() == [(1.0, "event")]


def test_format_and_print_queue(capsys):
    sched = HeapScheduler()
    sched.schedule(1.5, lambda: None, label="x->y")
    text = sched.format_queue()
    assert "1 event(s) pending" in text
    assert "t=1.5 : x->y" in text

    sched.print_queue()
    out = capsys.readouterr().out
    assert "t=1.5 : x->y" in out


def test_format_queue_empty():
    sched = HeapScheduler()
    assert "queue empty" in sched.format_queue()


def test_pending_reflects_time_remaining_during_run():
    sched = HeapScheduler()
    seen = {}

    def first():
        # schedule a follow-on, then snapshot what's left from inside a callback
        sched.schedule(5.0, lambda: None, label="later")
        seen["mid"] = sched.pending()

    sched.schedule(1.0, first, label="first")
    sched.schedule(2.0, lambda: None, label="second")
    sched.run()
    # at t=1 inside `first`: "second" (t=2) and the just-added "later" (t=6) remain
    assert seen["mid"] == [(2.0, "second"), (6.0, "later")]


# -- framework integration: meaningful labels --------------------------------
def test_framework_events_are_labelled():
    sched = HeapScheduler()
    system = RFSystem(sched)
    lna = system.add(Amplifier("LNA", gain_db=10.0, processing_delay=2.0))
    a = system.add(Recorder("recA"))
    b = system.add(Recorder("recB"))
    lna.subscribe(a)
    lna.subscribe(b)
    system.set_entry(lna)

    system.on_signal_rx(np.ones(8, dtype=np.complex64), 1e6, 1e9, at=5.0)
    # before running: the signalRX delivery is queued at t=5 with a label
    assert sched.pending() == [(5.0, "signalRX→LNA")]

    sched.run(until=5.0)  # fire the signalRX; LNA fans out to both recorders at t=7
    pend = sched.pending()
    assert pend == [(7.0, "LNA→recA"), (7.0, "LNA→recB")]
