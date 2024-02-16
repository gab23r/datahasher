import logging
import traceback
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

import ipyvuetify as v

import datahasher

T = TypeVar("T")


def log_error(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            func_name = getattr(func, "__name__", "func")
            datahasher.logger.warning(
                f"Impossible to execute {func_name}, fails with error : {e}"
            )
            for t in traceback.extract_tb(e.__traceback__):
                datahasher.logger.debug(
                    f"Error in {t.name} at line {t.lineno}: \n\t{t.line}"
                )

    return wrapper


def decorate_all_methods_with_log_error(cls: type) -> None:
    for name, method in cls.__dict__.items():
        if callable(method):
            setattr(cls, name, log_error(method))
        elif type(method) == property:
            decorated_property = property(
                fget=log_error(method.fget) if method.fget is not None else None,
                fset=log_error(method.fset) if method.fset is not None else None,
                fdel=log_error(method.fdel) if method.fdel is not None else None,
            )
            setattr(cls, name, decorated_property)


def log_error_class(original_class: type[T]) -> type[T]:
    # This function is used to decorate each method of a class with log_error
    orig_init = original_class.__init__
    # Make copy of original __init__, so we can call it without recursion

    def __init__(self: type, *args, **kws) -> None:
        decorate_all_methods_with_log_error(original_class)
        log_error(orig_init)(self, *args, **kws)  # Call the original __init__

    original_class.__init__ = __init__  # type: ignore # Set the class' __init__ to the new one

    return original_class


class Logger:
    """
    Each subclass of Logger will automatically have its methods decorated for log_error.

    Without this helper class we would need to decorate each classes with log_error_class.
    Instead we only need to inherite from Logger somehow

    """

    def __new__(cls, *args, **kwargs):
        decorate_all_methods_with_log_error(cls)
        return super().__new__(cls)


class OutputWidgetHandler(logging.Handler):
    """Custom logging handler sending logs to an output widget."""

    def __init__(self, logger_badge: v.Badge, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger_badge = logger_badge
        self.text_area = v.Textarea(
            v_model=None,
            rows=15,
            clearable=True,
            readonly=True,
            solo=True,
            hide_details=True,
        )

    def emit(self, record: logging.LogRecord) -> None:
        """Overload of logging.Handler method."""
        formatted_record = self.format(record) + "\n"
        if self.text_area.v_model is None:
            self.text_area.v_model = ""
        self.text_area.v_model = formatted_record + self.text_area.v_model

        self.logger_badge.v_model = True

        print(formatted_record)
