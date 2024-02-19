import dema.front


class Waiter:
    def __enter__(self):
        dema.front.app.linear_progess.show()
        dema.front.app.logger_bottom_sheet.v_model = True
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):  # noqa: ANN001
        dema.front.app.linear_progess.hide()
        if exception_type is None:
            dema.front.app.logger_bottom_sheet.v_model = False
            dema.front.app.app_bar.logger_badge.v_model = False
