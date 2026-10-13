"""Qt jobs: expensive work emits data; only the window updates widgets."""
import threading
import time
from PyQt5.QtCore import QThread, pyqtSignal


class BackgroundJob(QThread):
    completed = pyqtSignal(object)
    progress = pyqtSignal(object)

    def __init__(self, operation, parent=None):
        super().__init__(parent)
        self.operation = operation
        self.cancelled = threading.Event()

    def run(self):
        started = time.perf_counter()
        try:
            value = self.operation(self.cancelled, self.progress.emit)
            payload = {"result": value, "error": None}
        except Exception as error:
            payload = {"result": None, "error": str(error)}
        payload.update(job=self, seconds=time.perf_counter() - started,
                       cancelled=self.cancelled.is_set())
        self.completed.emit(payload)

    def cancel(self):
        self.cancelled.set()
