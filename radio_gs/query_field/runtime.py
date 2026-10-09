"""Standalone public entry point for the native-evidence/shared-member program."""
from .query_program import PROGRAM, query, category_partition
from .storage import load_field


class QueryFieldRuntime:
    """One scene export, current query evidence, and a fixed typed query program.

    NativeQueryEncoder produces text/image packets. The registered point
    adapters can consume this object directly. No source or observation loader
    is part of this runtime; manifest provenance is descriptive only.
    """
    program = PROGRAM

    def __init__(self, field, manifest=None):
        self.field = field
        self.manifest = manifest

    @classmethod
    def load(cls, directory, device="cpu"):
        return cls(*load_field(directory, device))

    def __getattr__(self, name):
        # Geometry and native spaces are shared with the registered adapters.
        return getattr(self.field, name)

    def query(self, packet):
        return query(self.field, packet)

    def category_partition(self, native_queries):
        return category_partition(self.field, native_queries)

    def text_union(self, native_query, generic_negatives, **options):
        from .query_program import text_union
        return text_union(self.field, native_query, generic_negatives, **options)

    def signed_regions(self, packet, **options):
        from .signed_regions import query
        return query(self.field, packet, **options)

    def positive_rays(self, packet, **options):
        from .positive_rays import query
        return query(self.field, packet, **options)
