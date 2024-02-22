from dema.front.logger import Logger, log_error_class # type: ignore
from ipyvuetable import EditingTable, Table

# def log_error_class(d):  # noqa: F811
#     return d  # type: ignore

class TableManager(log_error_class(Table), Logger):
    ...


class EditingTableManager(log_error_class(EditingTable), Logger):
    ...
