"""Capability-compatible encoders without invalidating sealed legacy sources."""
from radio_gs.models.native_query_encoders import NativeThreeQueryEncoder
from radio_gs.models.open_clip_native import OpenClipNativeRuntime


class FunctionalThreeQueryEncoder(NativeThreeQueryEncoder):
    def __init__(self, identity, device='cuda:0'):
        super().__init__(identity, device)
        self.backend = identity.get('semantic_backend', 'siglip2')
        if self.backend not in ('siglip2', 'open_clip'):
            raise ValueError('unsupported semantic encoder capability provider')

    def _clip(self):
        if self.vision is None:
            self.vision = OpenClipNativeRuntime(self.identity['open_clip'], self.device)
        return self.vision

    def encode_text(self, phrases):
        return self._clip().encode_text(phrases) if self.backend == 'open_clip' else super().encode_text(phrases)

    def encode_image(self, crop):
        return self._clip().encode_images([crop]) if self.backend == 'open_clip' else super().encode_image(crop)
