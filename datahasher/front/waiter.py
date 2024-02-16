import datahasher.front


class Waiter:
    def __enter__(self):
        datahasher.front.app.linear_progess.show()
        datahasher.front.app.logger_bottom_sheet.v_model = True
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):  # noqa: ANN001
        datahasher.front.app.linear_progess.hide()
        if exception_type is None:
            datahasher.front.app.logger_bottom_sheet.v_model = False
            datahasher.front.app.app_bar.logger_badge.v_model = False
