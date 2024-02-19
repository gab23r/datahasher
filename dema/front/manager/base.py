from dema.front.logger import Logger, log_error_class
from ipyvuetable import EditingTable, Table


class TableManager(log_error_class(Table), Logger):
    ...


class EditingTableManager(log_error_class(EditingTable), Logger):
    ...
