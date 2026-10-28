from f1pred.jolpica.ratelimit import SlidingWindowLimiter


class FakeTime:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


async def test_allows_burst_then_waits() -> None:
    t = FakeTime()
    limiter = SlidingWindowLimiter([(4, 1.0)], clock=t.clock, sleep=t.sleep)
    for _ in range(4):
        await limiter.acquire()
    assert t.sleeps == []
    await limiter.acquire()
    assert t.sleeps == [1.0]


async def test_enforces_longer_window_too() -> None:
    t = FakeTime()
    limiter = SlidingWindowLimiter([(4, 1.0), (6, 60.0)], clock=t.clock, sleep=t.sleep)
    for _ in range(6):
        await limiter.acquire()
    start = t.now
    await limiter.acquire()
    # The 7th call must wait until the first call leaves the 60 s window.
    assert t.now >= 60.0
    assert t.now > start
