.PHONY: docker-build test transfer toy two-rounds all clean-runs

IMAGE ?= rlxy-astrasim:0.6.0
RUN_ID ?= manual

docker-build:
	bash scripts/build_images.sh

test:
	./scripts/docker.sh test $(IMAGE)

transfer:
	./scripts/docker.sh transfer $(IMAGE) $(RUN_ID)

toy:
	./scripts/docker.sh toy $(IMAGE) $(RUN_ID)

two-rounds:
	./scripts/docker.sh two-rounds $(IMAGE) $(RUN_ID)

all:
	./scripts/docker.sh all $(IMAGE) $(RUN_ID)

clean-runs:
	@echo "Refusing to delete run evidence automatically. Remove a specific runs/<case>/<run_id> after review."

