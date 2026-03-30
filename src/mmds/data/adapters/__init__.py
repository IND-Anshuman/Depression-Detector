from __future__ import annotations

from .base import DatasetAdapter
from .daic_woz import DaicWozAdapter
from .edaic import EDaicAdapter
from .generic_video_folder import GenericVideoFolderAdapter

__all__ = ["DatasetAdapter", "DaicWozAdapter", "EDaicAdapter", "GenericVideoFolderAdapter"]
