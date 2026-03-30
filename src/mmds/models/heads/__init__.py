from __future__ import annotations

from .bdd import BDDHead
from .binary import BinaryHead
from .ordinal import OrdinalCoralHead
from .regression import RegressionHead

__all__ = ["BinaryHead", "OrdinalCoralHead", "RegressionHead", "BDDHead"]
