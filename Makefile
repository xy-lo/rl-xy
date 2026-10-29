.PHONY: docker-build test check transfer toy two-rounds all clean-runs

IMAGE ?= rlxy-astrasim:0.9.5
RUN_ID ?= $(shell date -u +%Y%m%dT%H%M%SZ)

docker-build:
	bash scripts/build_images.sh

test:
	docker run --rm --network none --memory 8g --memory-swap 8g --cpus 4 --pids-limit 512 \
	  --user "$$(id -u):$$(id -g)" --volume "$(CURDIR):/workspace/rl-xy:ro" \
	  --workdir /workspace/rl-xy --env PYTHONDONTWRITEBYTECODE=1 \
	  --env PYTHONPATH=/workspace/rl-xy/src:/opt/astra-sim --env ASTRA_SIM_ROOT=/opt/astra-sim \
	  --env ASTRA_SIM_BIN=/opt/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware \
	  --entrypoint python3 $(IMAGE) -m unittest discover -s tests -v

check:
	python3 scripts/check_public_results.py
	python3 scripts/check_qwen3_results.py
	python3 scripts/check_docs.py

transfer:
	./scripts/docker.sh transfer $(IMAGE) $(RUN_ID)

toy:
	./scripts/docker.sh toy $(IMAGE) $(RUN_ID)

two-rounds:
	./scripts/docker.sh two-rounds $(IMAGE) $(RUN_ID)

all:
	./scripts/docker.sh all $(IMAGE) $(RUN_ID)

clean-runs:
	@echo "Refusing to delete run evidence automatically. Review and remove only a specific owned run directory."
