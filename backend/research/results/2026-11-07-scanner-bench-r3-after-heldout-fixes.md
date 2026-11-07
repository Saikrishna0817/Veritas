# Scanner benchmark: r3-after-heldout-fixes / dev

371 generated malicious files, 147 benign files (public Hugging Face models at pinned revisions plus locally trained models). Files are scanned, never loaded.

| scanner | malicious caught | benign flagged |
|---|---|---|
| veritas (DANGEROUS) | 368/371 (99%) | 1/147 (1%) |
| veritas (DANGEROUS or REVIEW) | 371/371 (100%) | 5/147 (3%) |
| picklescan | 177/371 (48%) | 1/147 (1%) |
| modelscan | 135/371 (36%) | 1/147 (1%) |

Total scan time over the corpus (s): veritas 1.8, picklescan 0.8, modelscan 0.9

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
- **veritas (DANGEROUS or REVIEW)**: hf__nateraw__iris-svc__sklearn_model.joblib, hf__scikit-learn-examples__example__sklearn_model.joblib, hf__imodels__figs-compas-recidivism__sklearn_model.joblib, hf__Bingsu__adetailer__deepfashion2_yolov8s-seg.pt, hf__pyannote__wespeaker-voxceleb-resnet34-LM__pytorch_model.bin
- **picklescan**: hf__Bingsu__adetailer__deepfashion2_yolov8s-seg.pt
- **modelscan**: hf__Bingsu__adetailer__deepfashion2_yolov8s-seg.pt

# Scanner benchmark: r3-after-heldout-fixes / heldout

339 generated malicious files, 74 benign files (public Hugging Face models at pinned revisions plus locally trained models). Files are scanned, never loaded.

| scanner | malicious caught | benign flagged |
|---|---|---|
| veritas (DANGEROUS) | 339/339 (100%) | 0/74 (0%) |
| veritas (DANGEROUS or REVIEW) | 339/339 (100%) | 16/74 (22%) |
| picklescan | 68/339 (20%) | 0/74 (0%) |
| modelscan | 53/339 (16%) | 0/74 (0%) |

Total scan time over the corpus (s): veritas 1.8, picklescan 0.8, modelscan 0.9

## By payload family (caught/total)

| family | veritas (DANGEROUS) | veritas (DANGEROUS or REVIEW) | picklescan | modelscan |
|---|---|---|---|---|
| alias_sklearn_os | 7/7 | 7/7 | 0/7 | 0/7 |
| alias_sklearn_subprocess | 7/7 | 7/7 | 0/7 | 0/7 |
| atexit_register | 7/7 | 7/7 | 0/7 | 0/7 |
| atexit_unregister | 7/7 | 7/7 | 0/7 | 0/7 |
| codecs_open | 7/7 | 7/7 | 0/7 | 0/7 |
| dotted_numpy_os | 7/7 | 7/7 | 0/7 | 0/7 |
| dotted_pandas_os | 7/7 | 7/7 | 0/7 | 0/7 |
| dotted_scipy_builtins | 7/7 | 7/7 | 0/7 | 0/7 |
| dotted_sklearn_os | 7/7 | 7/7 | 0/7 | 0/7 |
| dotted_sklearn_subprocess | 7/7 | 7/7 | 0/7 | 0/7 |
| dotted_torch_sys | 7/7 | 7/7 | 0/7 | 0/7 |
| ensurepip | 7/7 | 7/7 | 0/7 | 0/7 |
| fileinput | 7/7 | 7/7 | 0/7 | 0/7 |
| ftplib_ftp | 7/7 | 7/7 | 0/7 | 0/7 |
| gc_objects | 7/7 | 7/7 | 0/7 | 0/7 |
| glob_glob | 7/7 | 7/7 | 0/7 | 0/7 |
| http_client | 7/7 | 7/7 | 0/7 | 0/7 |
| inspect_getmembers | 7/7 | 7/7 | 0/7 | 0/7 |
| io_open | 7/7 | 7/7 | 0/7 | 0/7 |
| memoized_global | 5/5 | 5/5 | 4/5 | 4/5 |
| mmap | 7/7 | 7/7 | 0/7 | 0/7 |
| os_execl | 7/7 | 7/7 | 5/7 | 5/7 |
| os_kill | 7/7 | 7/7 | 5/7 | 5/7 |
| os_remove | 7/7 | 7/7 | 5/7 | 5/7 |
| os_spawnlp | 7/7 | 7/7 | 5/7 | 5/7 |
| pathlib_path | 7/7 | 7/7 | 0/7 | 0/7 |
| proto5_frame | 5/5 | 5/5 | 4/5 | 4/5 |
| requests_get | 7/7 | 7/7 | 0/7 | 0/7 |
| resource_setrlimit | 7/7 | 7/7 | 0/7 | 0/7 |
| shelve_open | 7/7 | 7/7 | 0/7 | 0/7 |
| shutil_copy | 7/7 | 7/7 | 5/7 | 5/7 |
| shutil_move | 7/7 | 7/7 | 5/7 | 5/7 |
| signal_signal | 7/7 | 7/7 | 0/7 | 0/7 |
| smtplib_smtp | 7/7 | 7/7 | 0/7 | 0/7 |
| socket_socket | 7/7 | 7/7 | 5/7 | 5/7 |
| sqlite3_connect | 7/7 | 7/7 | 0/7 | 0/7 |
| ssl_wrap | 7/7 | 7/7 | 5/7 | 0/7 |
| subprocess_getoutput | 7/7 | 7/7 | 5/7 | 5/7 |
| sys_modules_pop | 7/7 | 7/7 | 5/7 | 5/7 |
| tarfile_open | 7/7 | 7/7 | 0/7 | 0/7 |
| tempfile_named | 7/7 | 7/7 | 0/7 | 0/7 |
| threading_thread | 7/7 | 7/7 | 0/7 | 0/7 |
| torch_load_from_bytes | 7/7 | 7/7 | 0/7 | 0/7 |
| trace | 7/7 | 7/7 | 0/7 | 0/7 |
| types_function | 7/7 | 7/7 | 0/7 | 0/7 |
| urllib_urlopen | 7/7 | 7/7 | 5/7 | 0/7 |
| venv_create | 7/7 | 7/7 | 5/7 | 0/7 |
| xmlrpc | 7/7 | 7/7 | 0/7 | 0/7 |
| zipfile_zipfile | 7/7 | 7/7 | 0/7 | 0/7 |

## By container variant (caught/total)

| variant | veritas (DANGEROUS) | veritas (DANGEROUS or REVIEW) | picklescan | modelscan |
|---|---|---|---|---|
| appended_after_clean | 49/49 | 49/49 | 14/49 | 11/49 |
| gzip_joblib | 49/49 | 49/49 | 0/49 | 0/49 |
| raw_p0 | 49/49 | 49/49 | 14/49 | 11/49 |
| raw_p4 | 49/49 | 49/49 | 14/49 | 11/49 |
| torch_zip | 47/47 | 47/47 | 12/47 | 9/47 |
| trojaned_sklearn | 49/49 | 49/49 | 14/49 | 11/49 |
| xz_wrapped | 47/47 | 47/47 | 0/47 | 0/47 |

## Benign files flagged

- **veritas (DANGEROUS)**: none
- **veritas (DANGEROUS or REVIEW)**: hf__kevinkyi__Homework2_Classical_ML__autogluon_predictor.pkl, hf__osanseviero__tips__clf.pkl, hf__osanseviero__tipsuhtxfu-sex-classification__clf.pkl, hf__maderix__titanic_traink4m62li8-survived-classification__clf.pkl, hf__merve__breast_cancernb8gjv4n-diagnosis-classification__clf.pkl, hf__srg__outhimar_64-Close-regression__clf.pkl, hf__luizapzbn__titanicht_mp88q-Survived-classification__clf.pkl, hf__cacauvicosa__heart1ohr2x9e-target-classification__clf.pkl, hf__Fulccrum__trainii_ac94u-label-classification__clf.pkl, hf__sahilrajpal121__train5a1e8w7-label-classification__clf.pkl, hf__ieborhan__irisg444_4c0-Species-classification__clf.pkl, hf__tehnlulz__pruned_datavq__ydnj-is_phishing-classification__clf.pkl, hf__robertbogdon__model_tuning_mindallee8kmcfjz-labels-classification__clf.pkl, hf__robertbogdon__model_tuning_mindalleeu83oz7r-labels-classification__clf.pkl, hf__robertbogdon__model_tuning_mindalle9_jsy6zj-labels-classification__clf.pkl, hf__Testys__diabetespmxrsn1x-diabetes-classification__clf.pkl
- **picklescan**: none
- **modelscan**: none
