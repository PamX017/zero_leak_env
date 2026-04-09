import openenv.core
import inspect

print("Contents of openenv.core:")
for name, obj in inspect.getmembers(openenv.core):
    if inspect.isclass(obj):
        print(f"Class: {name}")

try:
    from openenv.core import EnvClient
    print("Found EnvClient")
except ImportError:
    print("EnvClient not found in core")

try:
    from openenv.core.client import EnvClient
    print("Found EnvClient in core.client")
except ImportError:
    print("EnvClient not found in core.client")
