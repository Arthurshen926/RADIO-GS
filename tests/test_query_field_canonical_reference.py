import unittest
import torch
from scripts.research.run_query_field_canonical_subset_reference import initializer_from_state, validate_cohort


class CanonicalReferenceTest(unittest.TestCase):
    def test_missing_source_is_native_fallback_not_empty(self):
        prior = torch.tensor([[.8, .6], [.1, .9]])
        state = {0: {'start_membership': torch.tensor([.2, .7])}}
        result = initializer_from_state(prior, state, [{'views': 2}, {'views': 0}])
        self.assertTrue(torch.equal(result[:, 1], prior[:, 1]))
        self.assertTrue(torch.equal(result[:, 0], state[0]['start_membership']))
        with self.assertRaises(ValueError):
            initializer_from_state(prior, state, [{'views': 2}, {'views': 1}])

    def test_reject_development_duplicate_and_reordered_inputs(self):
        data = dict(texts=['a'], observed=torch.tensor([True, False]),
                    membership_by_control={'semantic_initial': torch.tensor([[.7], [0.]])})
        queries = {'queries': [{'text': 'a'}]}
        construction = {'source_records': [{'frame': i} for i in range(18)]}
        manifest = {'splits': {'train_frames': list(range(18)), 'development_frames': []}}
        self.assertEqual(validate_cohort(data, queries, construction, manifest), list(range(18)))
        manifest['splits']['development_frames'] = [5]
        with self.assertRaises(ValueError): validate_cohort(data, queries, construction, manifest)
        manifest['splits']['development_frames'] = []
        construction['source_records'][-1]['frame'] = 0
        with self.assertRaises(ValueError): validate_cohort(data, queries, construction, manifest)
        construction['source_records'][-1]['frame'] = 17
        with self.assertRaises(ValueError): validate_cohort(data, {'queries': [{'text': 'b'}]}, construction, manifest)


if __name__ == '__main__': unittest.main()
