"""Allow running the package as: python -m blindaid"""
import os
import sys
import warnings

# Suppress startup warnings and verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'  # Hide pygame message
os.environ['GLOG_minloglevel'] = '2'  # Suppress GLOG (oneDNN messages)
os.environ['PADDLEOCR_VERBOSE'] = '0'
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*pkg_resources.*')

# Redirect stderr temporarily to suppress PaddleOCR model messages
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

from blindaid.app import main

if __name__ == "__main__":
    main()
