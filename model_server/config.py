# 1. Model Path
MODEL_PATH = "/path/to/your/Model/xx"

# --- Optional Configurations ---

# 2. Server Settings
#   Host and port for the service to listen on.
HOST = "0.0.0.0"

# Define the port.
PORT = 8080 

# 3. GPU Configuration
#   Use accelerate to automatically assign GPUs.
#   Please modify the max memory for each GPU according to your hardware.
#   Example: {0: "30GiB", 1: "30GiB"} allocates a maximum of 30GiB to GPU 0 and GPU 1 respectively.
MAX_GPU_MEMORY = {0: "30GiB", 1: "30GiB"}
