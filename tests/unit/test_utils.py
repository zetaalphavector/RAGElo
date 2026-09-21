import pytest

from ragelo.utils import call_async_fn


async def double(value: int) -> int:
    return 2 * value


class TestCallAsyncFn:
    def test_runs_a_coroutine_to_completion(self):
        assert call_async_fn(double, 21) == 42

    def test_a_synchronous_wrapper_called_from_async_code_raises_instead_of_waiting_on_itself(self):
        async def calls_the_wrapper() -> int:
            return call_async_fn(double, 21)

        with pytest.raises(RuntimeError, match="Await the async method instead"):
            call_async_fn(calls_the_wrapper)
