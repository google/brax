from watchdog.events import FileSystemEventHandler
from rollout import append_unroll

class MjUnrollHandler(FileSystemEventHandler):
    """Handles new .mj_unroll files appearing in the base directory."""
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".mj_unroll"):
            append_unroll(event.src_path)
