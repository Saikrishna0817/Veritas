import sys
sys.path.insert(0, '.')
try:
    from app.demo.data_generator import get_demo_data, refresh_demo_data
    print('data_generator OK')
    from app.demo.real_datasets import DATASET_CATALOG
    print(f'real_datasets OK: {len(DATASET_CATALOG)} datasets')
    for d in DATASET_CATALOG:
        print(f'  - {d["id"]}')
    from app.api.routes import reports, datasets, models, upload
    print('all routes OK')
    print('SUCCESS')
except Exception as e:
    import traceback
    traceback.print_exc()
