import os
from stems.environment import STEMSEnvironment
base = os.path.expandvars(r"%LOCALAPPDATA%\\citylearn\\citylearn\\Cache\\v2.6.0b1\\datasets")
if not os.path.isdir(base):
    print("NO_BASE", base)
    raise SystemExit(0)
schemas = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])
print("SCHEMA_COUNT", len(schemas))
for s in schemas:
    try:
        e = STEMSEnvironment(schema=s, seed=0)
        print(f"{s}|B={e.num_buildings}|A={e.action_dim}|O={e.obs_dim}|mock={e.using_mock}")
    except Exception as ex:
        msg = str(ex).replace("\n", " ")[:160]
        print(f"{s}|ERROR|{msg}")
