from ipyvuetable import EditingTable, Table
from datahasher.front.logger import Logger, log_error_class


class TableManager(log_error_class(Table), Logger):
    ...


class EditingTableManager(log_error_class(EditingTable), Logger):
    ...
