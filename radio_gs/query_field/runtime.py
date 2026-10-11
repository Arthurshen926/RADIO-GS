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

    def prepare_text_extent(self, *, scope_device='cuda'):
        """Initialize the explicit text extent path before a file-denied query."""
        from .extent_runtime import ExtentTextRuntime
        self._extent_text_runtime=ExtentTextRuntime(self.field,scope_device=scope_device)
        return self._extent_text_runtime

    def text_extent(self, native_query, generic_negatives, *, scope_rule='atomic'):
        """One actual set, with no dependency on research-run predictions."""
        runtime=self.__dict__.get('_extent_text_runtime')
        if runtime is None:runtime=self.prepare_text_extent()
        return runtime.query(native_query,generic_negatives,scope_rule=scope_rule)

    def signed_regions(self, packet, **options):
        from .signed_regions import query
        return query(self.field, packet, **options)

    def positive_rays(self, packet, **options):
        from .positive_rays import query
        return query(self.field, packet, **options)

    def reference_scope(self, packet, camera, reference, **options):
        from .reference_scope import query
        return query(self.field, packet, camera, reference, **options)

    def reference_extent(self, packet, camera, reference, **options):
        """Shared actual-set decisions from the explicitly supplied current mask."""
        from .reference_extent import query
        return query(self.field, packet, camera, reference, **options)
