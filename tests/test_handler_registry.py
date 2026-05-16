"""Tests for the intent handler registry (C6)."""
from basketball_ai.chat.handlers import register_handler, get_handler
from basketball_ai.chat.intent import Intent


class TestHandlerRegistry:
    def test_register_and_get(self):
        @register_handler(Intent.HELP)
        def my_handler(ctx):
            return "custom help response"

        handler = get_handler(Intent.HELP)
        assert handler is not None
        assert handler({}) == "custom help response"

    def test_get_unregistered_returns_none(self):
        from basketball_ai.chat import handlers as _h
        _h._REGISTRY.pop(Intent.UNKNOWN, None)
        assert get_handler(Intent.UNKNOWN) is None

    def test_handler_overwrite(self):
        @register_handler(Intent.PREDICT)
        def first(ctx):
            return "first"

        @register_handler(Intent.PREDICT)
        def second(ctx):
            return "second"

        assert get_handler(Intent.PREDICT)({}) == "second"

    def test_handler_receives_context(self):
        received = {}

        @register_handler(Intent.TRAJECTORY)
        def capture(ctx):
            received.update(ctx)
            return "ok"

        get_handler(Intent.TRAJECTORY)({"intent": Intent.TRAJECTORY, "message": "test"})
        assert received["intent"] == Intent.TRAJECTORY
        assert received["message"] == "test"
