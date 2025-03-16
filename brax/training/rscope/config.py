from absl import flags
import sys
from pathlib import Path

# Parse command-line flags.
flags.DEFINE_string("logdir", "/tmp/rscope/active_run", "Path to the rscope directory.")
flags.FLAGS(sys.argv)

# Global paths.
BASE_PATH = Path(flags.FLAGS.logdir)
META_PATH = BASE_PATH / "rscope_meta.pkl"
