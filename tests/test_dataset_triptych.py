import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from analysis.plots import save_dataset_triptych
from data.schemas import DatasetBundle, TestResult


class DatasetTriptychTests(unittest.TestCase):
    def make_inputs(self, method='correlated_null_adaptive_permutation'):
        s = np.array([[0,0], [0,1], [1,0]], dtype=np.float32)
        a = np.array([[1,-1], [2,-2], [3,-3]], dtype=np.float32)
        ds = DatasetBundle(S=s, A=a)
        result = TestResult(method_name=method, metric='mse', p_value=1., stat_true=1.,
            stat_perm=np.array([2.,3.]), runtime_sec=0., n_cells=3, n_genes=2,
            config={}, artifacts={'lowest_S':s, 'highest_S':s,
                'lowest_A':a*2, 'highest_A':a*3, 'lowest_perm_index':0, 'highest_perm_index':1})
        return ds, result

    def test_expression_null_panels_use_selected_targets_and_shared_scale(self):
        ds, result = self.make_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'dataset.png'
            with patch('analysis.plots._plot_spatial_dataset_heatmap') as draw:
                save_dataset_triptych(ds, result, path)
            for i, call in enumerate(draw.call_args_list):
                np.testing.assert_array_equal(call.args[2], np.arange(1,4)*(i+1))
                self.assertEqual(call.kwargs['vmin'], 1.)
                self.assertEqual(call.kwargs['vmax'], 9.)
            with np.load(Path(tmp)/'dataset_inputs.npz') as saved:
                for key in ('lowest_A','highest_A'):
                    np.testing.assert_array_equal(saved[key],result.artifacts[key])
                np.testing.assert_array_equal(saved['true_A'],ds.A)
                self.assertEqual(int(saved['highest_perm_index']),1)

    def test_coordinate_null_reuses_expression_at_permuted_coordinates(self):
        ds, result = self.make_inputs('parallel_permutation')
        del result.artifacts['lowest_A']; del result.artifacts['highest_A']
        result.artifacts['lowest_S'] = ds.S[::-1]
        with tempfile.TemporaryDirectory() as tmp, patch('analysis.plots._plot_spatial_dataset_heatmap') as draw:
            save_dataset_triptych(ds,result,Path(tmp)/'dataset.png')
            for call in draw.call_args_list:
                np.testing.assert_array_equal(call.args[2],[1,2,3])
            np.testing.assert_array_equal(draw.call_args_list[1].args[1],ds.S[::-1])

    def test_missing_expression_null_targets_cannot_silently_replot_true(self):
        ds, result = self.make_inputs()
        del result.artifacts['lowest_A']
        with self.assertRaisesRegex(ValueError,'training targets'):
            save_dataset_triptych(ds,result,'unused.png')

if __name__=='__main__': unittest.main()
