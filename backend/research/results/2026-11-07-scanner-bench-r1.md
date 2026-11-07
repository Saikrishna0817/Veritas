# Scanner benchmark: r1

371 generated malicious files, 147 benign files (public Hugging Face models at pinned revisions plus locally trained models). Files are scanned, never loaded.

| scanner | malicious caught | benign flagged |
|---|---|---|
| veritas (DANGEROUS) | 368/371 (99%) | 1/147 (1%) |
| veritas (DANGEROUS or REVIEW) | 371/371 (100%) | 9/147 (6%) |
| picklescan | 177/371 (48%) | 1/147 (1%) |
| modelscan | 135/371 (36%) | 1/147 (1%) |

Total scan time over the corpus (s): veritas 1.1, picklescan 0.3, modelscan 0.5

## By payload family (caught/total)

| family | veritas (DANGEROUS) | veritas (DANGEROUS or REVIEW) | picklescan | modelscan |
|---|---|---|---|---|
| _pickle_loads | 7/7 | 7/7 | 5/7 | 5/7 |
| asyncio_shell | 7/7 | 7/7 | 5/7 | 0/7 |
| builtins_compile | 7/7 | 7/7 | 5/7 | 5/7 |
| builtins_eval | 7/7 | 7/7 | 5/7 | 5/7 |
| builtins_exec | 7/7 | 7/7 | 5/7 | 5/7 |
| builtins_import | 7/7 | 7/7 | 0/7 | 5/7 |
| builtins_open | 7/7 | 7/7 | 5/7 | 5/7 |
| code_interact | 7/7 | 7/7 | 0/7 | 0/7 |
| cprofile | 7/7 | 7/7 | 5/7 | 0/7 |
| ctypes | 7/7 | 7/7 | 5/7 | 0/7 |
| getattr_import_chain | 5/5 | 5/5 | 4/5 | 4/5 |
| importlib | 7/7 | 7/7 | 0/7 | 0/7 |
| inst_opcode | 5/5 | 5/5 | 4/5 | 4/5 |
| joblib_parallel | 7/7 | 7/7 | 0/7 | 0/7 |
| keras_lambda | 1/1 | 1/1 | 0/1 | 0/1 |
| marshal_loads | 7/7 | 7/7 | 0/7 | 0/7 |
| multiprocessing | 7/7 | 7/7 | 0/7 | 0/7 |
| npy_object_array | 1/1 | 1/1 | 1/1 | 0/1 |
| npz_object_array | 1/1 | 1/1 | 1/1 | 0/1 |
| nt_system | 7/7 | 7/7 | 5/7 | 5/7 |
| numpy_f2py | 7/7 | 7/7 | 5/7 | 0/7 |
| numpy_load | 7/7 | 7/7 | 0/7 | 0/7 |
| numpy_runstring | 7/7 | 7/7 | 5/7 | 0/7 |
| obj_opcode | 5/5 | 5/5 | 4/5 | 4/5 |
| onnx_custom_op | 0/1 | 1/1 | 0/1 | 0/1 |
| onnx_external_absolute | 1/1 | 1/1 | 0/1 | 0/1 |
| onnx_external_traversal | 1/1 | 1/1 | 0/1 | 0/1 |
| operator_methodcaller | 7/7 | 7/7 | 5/7 | 0/7 |
| os_execv | 7/7 | 7/7 | 5/7 | 5/7 |
| os_popen | 7/7 | 7/7 | 5/7 | 5/7 |
| os_system | 7/7 | 7/7 | 5/7 | 5/7 |
| pandas_eval | 7/7 | 7/7 | 0/7 | 0/7 |
| pandas_io_pickle | 7/7 | 7/7 | 0/7 | 0/7 |
| pandas_read_pickle | 7/7 | 7/7 | 0/7 | 0/7 |
| partial_wrapped | 5/5 | 5/5 | 4/5 | 4/5 |
| pickle_loads | 7/7 | 7/7 | 5/7 | 5/7 |
| pip_main | 7/7 | 7/7 | 5/7 | 0/7 |
| platform_popen | 7/7 | 7/7 | 0/7 | 0/7 |
| posix_system | 7/7 | 7/7 | 5/7 | 5/7 |
| pty_spawn | 7/7 | 7/7 | 5/7 | 5/7 |
| pydoc_pipepager | 7/7 | 7/7 | 5/7 | 0/7 |
| python2_eval | 7/7 | 7/7 | 5/7 | 5/7 |
| reduce_via_copyreg | 5/5 | 5/5 | 4/5 | 4/5 |
| runpy | 7/7 | 7/7 | 5/7 | 5/7 |
| safetensors_bad_offsets | 0/1 | 1/1 | 0/1 | 0/1 |
| safetensors_huge_header | 0/1 | 1/1 | 0/1 | 0/1 |
| shutil_rmtree | 7/7 | 7/7 | 5/7 | 5/7 |
| sklearn_run_script | 7/7 | 7/7 | 0/7 | 0/7 |
| socket | 7/7 | 7/7 | 5/7 | 5/7 |
| subprocess_call | 7/7 | 7/7 | 5/7 | 5/7 |
| subprocess_check_output | 7/7 | 7/7 | 5/7 | 5/7 |
| subprocess_popen | 7/7 | 7/7 | 5/7 | 5/7 |
| subprocess_run | 7/7 | 7/7 | 5/7 | 5/7 |
| sys_exit | 7/7 | 7/7 | 5/7 | 5/7 |
| tempfile_mkstemp | 7/7 | 7/7 | 0/7 | 0/7 |
| timeit | 7/7 | 7/7 | 5/7 | 0/7 |
| torch_hub | 7/7 | 7/7 | 0/7 | 0/7 |
| torch_jit_load | 7/7 | 7/7 | 0/7 | 0/7 |
| torch_load | 7/7 | 7/7 | 0/7 | 0/7 |
| torch_utils_cpp | 7/7 | 7/7 | 0/7 | 0/7 |
| webbrowser | 7/7 | 7/7 | 5/7 | 5/7 |
| zip_executable_member | 1/1 | 1/1 | 0/1 | 0/1 |
| zip_so_member | 1/1 | 1/1 | 0/1 | 0/1 |

## By container variant (caught/total)

| variant | veritas (DANGEROUS) | veritas (DANGEROUS or REVIEW) | picklescan | modelscan |
|---|---|---|---|---|
| appended_after_clean | 53/53 | 53/53 | 36/53 | 28/53 |
| gzip_joblib | 53/53 | 53/53 | 0/53 | 0/53 |
| keras | 1/1 | 1/1 | 0/1 | 0/1 |
| npy | 1/1 | 1/1 | 1/1 | 0/1 |
| npz | 1/1 | 1/1 | 1/1 | 0/1 |
| onnx | 2/3 | 3/3 | 0/3 | 0/3 |
| raw_p0 | 53/53 | 53/53 | 36/53 | 28/53 |
| raw_p4 | 53/53 | 53/53 | 36/53 | 28/53 |
| safetensors | 0/2 | 2/2 | 0/2 | 0/2 |
| torch_zip | 48/48 | 48/48 | 31/48 | 23/48 |
| trojaned_sklearn | 53/53 | 53/53 | 36/53 | 28/53 |
| xz_wrapped | 48/48 | 48/48 | 0/48 | 0/48 |
| zip | 2/2 | 2/2 | 0/2 | 0/2 |

## Benign files flagged

- **veritas (DANGEROUS)**: hf__Bingsu__adetailer__deepfashion2_yolov8s-seg.pt
- **veritas (DANGEROUS or REVIEW)**: sk_gridsearch.pkl, sk_gridsearch_p2.pkl, torchscript.pt, hf__nateraw__iris-svc__sklearn_model.joblib, hf__scikit-learn-examples__example__sklearn_model.joblib, hf__imodels__figs-compas-recidivism__sklearn_model.joblib, hf__Xenova__whisper-tiny.en__encoder_model_q4f16.onnx, hf__Bingsu__adetailer__deepfashion2_yolov8s-seg.pt, hf__pyannote__wespeaker-voxceleb-resnet34-LM__pytorch_model.bin
- **picklescan**: hf__Bingsu__adetailer__deepfashion2_yolov8s-seg.pt
- **modelscan**: hf__Bingsu__adetailer__deepfashion2_yolov8s-seg.pt
