import os

# torch and faiss-cpu each bundle libomp on macOS; loading both segfaults without these.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
