"""Read-only handles over existing region maps; no duplicate map materialization."""
from dataclasses import dataclass
import json
from pathlib import Path


def region_layout(offsets, source_frames, region_count):
    """Validate common region provenance without changing either normalization."""
    import numpy as np
    offsets = np.asarray(offsets)
    if (offsets.ndim != 1 or offsets.dtype.kind not in 'iu' or
            len(offsets) != len(source_frames)+1 or not len(offsets) or
            offsets[0] != 0 or offsets[-1] != region_count or (np.diff(offsets) < 0).any()):
        raise ValueError('region/view domains differ')
    if len(set(source_frames)) != len(source_frames):
        raise ValueError('duplicate source view identities')
    return np.repeat(np.arange(len(source_frames)), np.diff(offsets))


@dataclass(frozen=True)
class CompiledRegionEvidence:
    """Read-only view over an existing Gaussian-contribution region payload.

    'supported' means positive region contribution mass, not independent
    geometric visibility. A missing visibility relation is never invented.
    """
    payload: dict
    region_frames: object
    normalization: str = 'gaussian_contribution_mass_then_carrier_mass'
    independent_visibility_available: bool = False

    @classmethod
    def from_payload(cls, payload):
        import torch
        rows, columns = payload['shape']
        descriptor = payload['descriptors']
        if descriptor.ndim != 2 or len(descriptor) != columns or tuple(payload['supported'].shape) != (rows,):
            raise ValueError('descriptor/transfer/support domains differ')
        crow, index, weights = [payload[k] for k in ('crow','columns','weights')]
        if (crow.ndim != 1 or index.ndim != 1 or weights.ndim != 1 or len(crow) != rows+1 or
                int(crow[0]) != 0 or int(crow[-1]) != len(index) or len(index) != len(weights) or
                bool((crow[1:] < crow[:-1]).any()) or bool(((index < 0) | (index >= columns)).any())):
            raise ValueError('invalid sparse region contribution indices')
        if (crow.dtype not in (torch.int32,torch.int64) or index.dtype not in (torch.int32,torch.int64) or
                payload['supported'].dtype != torch.bool or
                not bool(torch.isfinite(weights).all()) or bool((weights < 0).any())):
            raise ValueError('invalid sparse region contribution values')
        # Early compiled fields retained only the normalized operator. Missing
        # view provenance must remain explicit, not be guessed from scene order.
        has_offsets = 'region_offsets' in payload
        has_frames = 'source_frames' in payload
        if has_offsets != has_frames:
            raise ValueError('partial region provenance is not a complete binding')
        frames = (region_layout(payload['region_offsets'].cpu().numpy(), payload['source_frames'], columns)
                  if has_offsets else None)
        visibility = payload.get('geometric_visibility')
        if visibility is not None:
            if frames is None:
                raise ValueError('geometric visibility requires source view provenance')
            if (list(visibility['shape']) != [rows, len(payload['source_frames'])]
                    or visibility.get('kind') != 'unconditioned_compositing_mass'
                    or len(visibility.get('raster_hw', [])) != len(payload['source_frames'])
                    or any(len(hw) != 2 or any(type(x) is not int or x <= 0 for x in hw)
                           for hw in visibility['raster_hw'])):
                raise ValueError('invalid geometric visibility contract')
            vc, vi, vw = [visibility[k] for k in ('crow', 'columns', 'weights')]
            if (vc.ndim != 1 or vi.ndim != 1 or vw.ndim != 1 or len(vc) != rows+1
                    or vc.dtype not in (torch.int32, torch.int64) or vi.dtype not in (torch.int32, torch.int64)
                    or int(vc[0]) != 0 or int(vc[-1]) != len(vi) or len(vi) != len(vw)
                    or bool((vc[1:] < vc[:-1]).any())
                    or bool(((vi < 0) | (vi >= len(payload['source_frames']))).any())
                    or not bool(torch.isfinite(vw).all()) or bool((vw <= 0).any())):
                raise ValueError('invalid geometric visibility relation')
        return cls(payload, frames, independent_visibility_available=visibility is not None)

    def transfer(self, device):
        import torch
        from .core import _record
        p = self.payload
        _record('compiled_region_store')
        return torch.sparse_csr_tensor(p['crow'].to(device),p['columns'].to(device),
                                      p['weights'].to(device),size=p['shape'])

    def visibility(self, device):
        """Unconditioned compositing mass, or None; never region support as a fallback."""
        import torch
        if not self.independent_visibility_available:
            return None
        p = self.payload['geometric_visibility']
        return torch.sparse_csr_tensor(p['crow'].to(device), p['columns'].to(device),
                                      p['weights'].to(device), size=p['shape'])


@dataclass(frozen=True)
class EvidenceStore:
    manifest: Path
    specification: dict

    @classmethod
    def open(cls, manifest):
        path = Path(manifest).resolve()
        spec = json.loads(path.read_text())
        return cls(path, spec)

    def mapped_assets(self):
        """Resolve only the explicitly declared registered-map assets."""
        result = {}
        root = self.manifest.parent.resolve()
        for name in self.specification.get('map_files', []):
            path = (root / name).resolve()
            if root not in path.parents:
                raise ValueError('map asset escapes its declared root')
            if not path.is_file():
                raise FileNotFoundError(path)
            result[name] = path
        return result

    def registered_relations(self):
        from scipy import sparse
        import numpy as np
        assets = self.mapped_assets()
        membership = sparse.load_npz(assets['membership.npz'])
        visibility = sparse.load_npz(assets['visibility.npz'])
        offsets = np.load(assets['region_offsets.npy'], mmap_mode='r')
        if membership.shape[0] != visibility.shape[0]:
            raise ValueError('membership/visibility carrier differs')
        source_frames = self.specification.get('source_frames', list(range(visibility.shape[1])))
        if len(source_frames) != visibility.shape[1]:
            raise ValueError('region/view domains differ')
        frames = region_layout(offsets, source_frames, membership.shape[1])
        return membership, visibility, frames
