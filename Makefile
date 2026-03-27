PY ?= python
CFG ?= configs/kg_math.yaml

.PHONY: all seed msc concepts nodes pre sem

all: seed msc concepts nodes pre sem

seed:
	$(PY) scripts/01_prepare_math_seed.py --config $(CFG)

msc:
	$(PY) scripts/00_build_msc_catalog.py --config $(CFG)

concepts:
	$(PY) scripts/02_extract_concepts.py --config $(CFG)

nodes:
	$(PY) scripts/03_build_nodes_and_seeds.py --config $(CFG)

pre:
	$(PY) scripts/04_build_edges_pre.py --config $(CFG)

sem:
	$(PY) scripts/05_build_edges_sem.py --config $(CFG)
