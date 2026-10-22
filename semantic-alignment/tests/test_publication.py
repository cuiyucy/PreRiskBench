"""Synthetic-only tests against copied research functions and safety wrappers."""
import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'pipelines/benchmark/src'))
from release_runtime import expand_path, expand_config, require_patient_map, verify_patient_disjoint

class Interfaces(unittest.TestCase):
    def test_environment_expands(self):
        with patch.dict(os.environ,{'SYNTHETIC_DATA_ROOT':'/tmp/synthetic'}):
            self.assertEqual(expand_path('${SYNTHETIC_DATA_ROOT}/x'),'/tmp/synthetic/x')
    def test_missing_environment_fails(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaises(ValueError): expand_path('${MISSING_TEST_PATH}/x')
    def test_nested_config_expands(self):
        with patch.dict(os.environ,{'SYNTHETIC_VALUE':'abc'}):
            self.assertEqual(expand_config({'x':['${SYNTHETIC_VALUE}']}),{'x':['abc']})
    def test_no_surgery_fallback(self):
        with self.assertRaises(ValueError): require_patient_map(['SYN_S1'],{})
        with self.assertRaises(ValueError): require_patient_map(['SYN_S1','SYN_S2'],{'SYN_S1':'SYN_P1'})
    def test_leakage_detected(self):
        with self.assertRaises(ValueError): verify_patient_disjoint({'train':['S1'],'test':['S2']},{'S1':'P1','S2':'P1'})
    def test_plan_does_not_execute(self):
        run=subprocess.run([sys.executable,'-B',str(ROOT/'release.py'),'benchmark','--','run','--exp','baseline'],capture_output=True,text=True)
        self.assertEqual(run.returncode,0)
        self.assertIn('PLAN ONLY',run.stdout)
    def test_other_task_rejected(self):
        run=subprocess.run([sys.executable,'-B',str(ROOT/'release.py'),'benchmark','--','run','--task','icu_admit'],capture_output=True,text=True)
        self.assertNotEqual(run.returncode,0)
    def test_original_benchmark_help(self):
        run=subprocess.run([sys.executable,'-B',str(ROOT/'release.py'),'benchmark','--','--help'],capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)
        self.assertIn('build',run.stdout)

class ResearchFunctions(unittest.TestCase):
    def _llm_constructor(self):
        """Exercise the real constructor without optional models, files or API calls."""
        import threading
        from types import SimpleNamespace
        from typing import Dict
        path=ROOT/'pipelines/alignment/src/snomed_mapping/3.1.llm_Mapping_processor.py'
        tree=ast.parse(path.read_text(encoding='utf-8'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DeepSeekBatchProcessorV3')
        init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        api=SimpleNamespace(OpenAI=Mock())
        scope={'Dict':Dict,'os':SimpleNamespace(path=os.path,makedirs=Mock()),
               'threading':threading,'ProcessingStats':Mock(),'openai':api,
               'HAS_VECTOR':False,'HAS_TRANSFORMER':False}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[init],type_ignores=[])),str(path),'exec'),scope)
        instance=SimpleNamespace(_setup_logging=Mock())
        return scope['__init__'],instance,api
    def test_llm_uses_configured_endpoint_without_network(self):
        init,instance,api=self._llm_constructor()
        init(instance,{'llm':{'api_url':'https://example.invalid/v1'},'rag':{'enabled':False}},
             'synthetic.csv',output_dir='synthetic-output')
        self.assertEqual(api.OpenAI.call_args.kwargs['base_url'],'https://example.invalid/v1')
        self.assertIs(instance.client,api.OpenAI.return_value)
    def test_llm_requires_explicit_endpoint_without_network(self):
        init,instance,api=self._llm_constructor()
        with self.assertRaisesRegex(ValueError,'llm.api_url'):
            init(instance,{'rag':{'enabled':False}},'synthetic.csv',output_dir='synthetic-output')
        api.OpenAI.assert_not_called()
    def test_frequency_cluster_selector(self):
        path=ROOT/'pipelines/benchmark/src/analysis/clustering_pipeline.py'
        tree=ast.parse(path.read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='select_centroid_by_frequency')
        scope={}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(path),'exec'),scope)
        choose=scope['select_centroid_by_frequency']
        self.assertEqual(choose([0,1],['synthetic_a','synthetic_b'],
            {'synthetic_a':{'SYN_A'},'synthetic_b':{'SYN_B'}},{'SYN_A':1,'SYN_B':9}),('synthetic_b','SYN_B'))
        self.assertEqual(choose([0],['synthetic'],{'synthetic':{'B','A'}},{}),('synthetic','A'))
    def test_actual_patient_grouped_sampler(self):
        import pandas as pd
        from data.build_benchmark import StratifiedSampler
        rows=[]; mapping={}
        for patient in range(250):
            for operation in range(2):
                sid=f'SYN_S{patient}_{operation}'
                rows.append({'surgery_id':sid,'asa_label':patient%5+1})
                mapping[sid]=f'SYN_P{patient}'
        frame=pd.DataFrame(rows)
        require_patient_map(frame['surgery_id'],mapping)
        sampler=StratifiedSampler(random_state=42)
        first=sampler.split(frame,'asa','synthetic',patient_map=mapping)
        second=sampler.split(frame,'asa','synthetic',patient_map=mapping)
        self.assertEqual(first,second)
        self.assertTrue(verify_patient_disjoint(first,mapping))
        self.assertEqual(sum(map(len,first.values())),500)
    def test_actual_identifier_drop(self):
        import pandas as pd
        from data.preprocessor import DataPreprocessor
        frame=pd.DataFrame({'patient_id':['SYN_P'],'surgery_id':['SYN_S'],'mrn':['SYN_M'],'age':[55]})
        self.assertEqual(DataPreprocessor().remove_id_and_summary_columns(frame).columns.tolist(),['age'])
    def test_actual_sampler_rejects_missing_patient_map(self):
        import pandas as pd
        from data.build_benchmark import StratifiedSampler
        with self.assertRaises(ValueError):
            StratifiedSampler().split(pd.DataFrame({'surgery_id':['SYN_S'],'asa_label':[1]}),'asa','synthetic')
    def test_actual_runner_config_and_imports(self):
        from experiments.run_experiments import ExperimentRunner
        runner=ExperimentRunner(seed=42)
        self.assertEqual(runner.config['data']['target_tasks'],['asa'])
    def test_synthetic_metric(self):
        from sklearn.metrics import f1_score
        self.assertEqual(f1_score([1,2,3,4,5],[1,2,3,4,5],average='macro'),1.0)

if __name__=='__main__': unittest.main()
