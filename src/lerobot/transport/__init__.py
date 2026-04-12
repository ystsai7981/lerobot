"""
gRPC transport layer for async inference.

Requires: ``pip install 'lerobot[grpcio-dep]'``

Available modules (import directly)::

    from lerobot.transport.utils import ...
"""

from lerobot.utils.import_utils import require_package

require_package("grpcio", extra="grpcio-dep", import_name="grpc")

__all__: list[str] = []
