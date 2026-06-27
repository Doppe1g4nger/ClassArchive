import pytest

from rfdes import HeapScheduler


def test_events_fire_in_time_order():
    sched = HeapScheduler()
    fired = []
    sched.schedule(3.0, lambda: fired.append("c"))
    sched.schedule(1.0, lambda: fired.append("a"))
    sched.schedule(2.0, lambda: fired.append("b"))
    sched.run()
    assert fired == ["a", "b", "c"]


def test_same_timestamp_is_fifo():
    sched = HeapScheduler()
    fired = []
    for i in range(5):
        sched.schedule(1.0, lambda i=i: fired.append(i))
    sched.run()
    assert fired == [0, 1, 2, 3, 4]


def test_now_advances_to_fired_event_time():
    sched = HeapScheduler()
    seen = []
    sched.schedule(2.5, lambda: seen.append(sched.now()))
    sched.run()
    assert seen == [2.5]


def test_cancel_prevents_firing():
    sched = HeapScheduler()
    fired = []
    sched.schedule(1.0, lambda: fired.append("a"))
    handle = sched.schedule(2.0, lambda: fired.append("b"))
    sched.cancel(handle)
    sched.run()
    assert fired == ["a"]


def test_run_until_leaves_later_events_queued():
    sched = HeapScheduler()
    fired = []
    sched.schedule(1.0, lambda: fired.append("a"))
    sched.schedule(5.0, lambda: fired.append("b"))
    sched.run(until=3.0)
    assert fired == ["a"]
    assert sched.now() == 1.0
    # the later event is still queued and runs on a subsequent run
    sched.run()
    assert fired == ["a", "b"]
    assert sched.now() == 5.0


def test_relative_delays_accumulate():
    sched = HeapScheduler()
    times = []

    def first():
        times.append(sched.now())
        sched.schedule(2.0, second)  # relative to current time (1.0)

    def second():
        times.append(sched.now())

    sched.schedule(1.0, first)
    sched.run()
    assert times == [1.0, 3.0]


def test_negative_delay_rejected():
    sched = HeapScheduler()
    with pytest.raises(ValueError):
        sched.schedule(-1.0, lambda: None)
