"""Check how CUDA device ordinals map to nvidia-smi indices / UUIDs on this host.

Prints, for the default order and for CUDA_DEVICE_ORDER=PCI_BUS_ID, the UUID
that torch sees at each ordinal, next to nvidia-smi's index -> UUID table.
Each order is probed in a fresh subprocess because CUDA reads the variable at
initialisation.
"""
import json
import os
import subprocess
import sys

PROBE = r"""
import json, torch
out = []
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    out.append({"ordinal": i, "uuid": "GPU-" + str(p.uuid), "name": p.name,
                "pci_bus_id": getattr(p, "pci_bus_id", None)})
print(json.dumps(out))
"""


def probe(order):
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    if order:
        env["CUDA_DEVICE_ORDER"] = order
    else:
        env.pop("CUDA_DEVICE_ORDER", None)
    r = subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, check=True)
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader"],
        capture_output=True, text=True, check=True).stdout.strip().splitlines()
    smi_rows = [dict(zip(["index", "uuid", "pci"], [x.strip() for x in line.split(",")])) for line in smi]
    by_uuid = {r["uuid"]: r["index"] for r in smi_rows}
    result = {"nvidia_smi": smi_rows}
    for label, order in [("default", None), ("PCI_BUS_ID", "PCI_BUS_ID")]:
        rows = probe(order)
        for r in rows:
            r["nvidia_smi_index"] = by_uuid.get(r["uuid"])
        result[label] = rows
        same = all(str(r["ordinal"]) == r["nvidia_smi_index"] for r in rows)
        print(f"[{label}] ordinal == nvidia-smi index for all devices: {same}")
        for r in rows:
            print(f"   cuda:{r['ordinal']} -> {r['uuid']} (nvidia-smi {r['nvidia_smi_index']})")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
