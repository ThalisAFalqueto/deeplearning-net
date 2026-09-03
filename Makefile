setup:
	uv sync
	uv pip install torch torchvision --torch-backend=auto

setup-cpu:
	uv sync
	uv pip install torch torchvision --torch-backend=cpu