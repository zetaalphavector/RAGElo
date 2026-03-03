import logging

from rich.logging import RichHandler

from ragelo.logger import configure_logging


class TestConfigureLogging:
    def setup_method(self):
        """Reset the ragelo logger state before each test."""
        lib_logger = logging.getLogger("ragelo")
        lib_logger.handlers = [logging.NullHandler()]
        lib_logger.setLevel(logging.WARNING)

    def test_sets_level(self):
        configure_logging(level="DEBUG", rich=False)
        assert logging.getLogger("ragelo").level == logging.DEBUG

    def test_sets_level_from_int(self):
        configure_logging(level=logging.INFO, rich=False)
        assert logging.getLogger("ragelo").level == logging.INFO

    def test_rich_handler_by_default(self):
        configure_logging(level="INFO")
        lib_logger = logging.getLogger("ragelo")
        non_null = [h for h in lib_logger.handlers if not isinstance(h, logging.NullHandler)]
        assert len(non_null) == 1
        assert isinstance(non_null[0], RichHandler)

    def test_stream_handler_when_rich_false(self):
        configure_logging(level="INFO", rich=False)
        lib_logger = logging.getLogger("ragelo")
        non_null = [h for h in lib_logger.handlers if not isinstance(h, logging.NullHandler)]
        assert len(non_null) == 1
        assert isinstance(non_null[0], logging.StreamHandler)
        assert not isinstance(non_null[0], RichHandler)

    def test_idempotent(self):
        configure_logging(level="INFO", rich=False)
        configure_logging(level="DEBUG", rich=False)
        lib_logger = logging.getLogger("ragelo")
        non_null = [h for h in lib_logger.handlers if not isinstance(h, logging.NullHandler)]
        assert len(non_null) == 1
        assert lib_logger.level == logging.DEBUG

    def test_null_handler_preserved(self):
        configure_logging(level="INFO", rich=False)
        lib_logger = logging.getLogger("ragelo")
        null_handlers = [h for h in lib_logger.handlers if isinstance(h, logging.NullHandler)]
        assert len(null_handlers) == 1

    def test_default_has_only_null_handler(self):
        lib_logger = logging.getLogger("ragelo")
        assert all(isinstance(h, logging.NullHandler) for h in lib_logger.handlers)
