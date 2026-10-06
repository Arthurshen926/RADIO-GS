"""Process boundaries for providers with distinct native/SAM environments.

Stable numerical stages are reused without importing historical query results.
Only the named stage is loaded, so native PyTorch and SAM3 stay isolated.
"""
import importlib
import sys

STAGES = {
    'native-text': 'radio_gs.pipeline.map_query',
    'detect-text': 'scripts.research.detect_consolidated_sources',
    'record-identity': 'scripts.research.filter_source_identity',
    'admit-identity': 'scripts.research.condition_source_identity',
    'resolve-reference': 'radio_gs.pipeline.resolve_reference',
    'track-reference': 'radio_gs.pipeline.tracking',
    'fit-reference': 'radio_gs.querying.reference_response',
    'compose-reference': 'scripts.research.compose_reference_readout',
    'fit-selection': 'radio_gs.pipeline.subset',
    'fit-response': 'radio_gs.pipeline.response',
}


def main():
    stage = sys.argv.pop(1)
    importlib.import_module(STAGES[stage]).main()


if __name__ == '__main__':
    main()
