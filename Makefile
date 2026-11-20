# One command per stage. `make synthetic` reproduces everything from scratch
# without the licensed data; `make real` needs data/real/raw/*.gz.
# Run `make help` for a summary.

IMAGE    ?= jse-microstructure
PLATFORM ?=
CORES    ?= 2
SNAKE     = snakemake -s workflow/Snakefile --cores $(CORES)

.PHONY: help test synthetic real pipeline-real pipeline-synthetic render render-synthetic \
        publish-local clean docker-build docker-synthetic docker-real docker-shell

help:                ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

test:                ## Unit tests (no data needed)
	python -m pytest -q

pipeline-synthetic:  ## Simulate raw exports and run the whole pipeline on them
	JSEMICRO_DATA=synthetic $(SNAKE)

pipeline-real:       ## Run the whole pipeline on the licensed exports in data/real/raw
	JSEMICRO_DATA=real $(SNAKE)

SYNTH_TMP ?= /tmp/jsemicro-synthetic-render

render-synthetic:    ## Render the book from the synthetic results (no freeze) into _book_synthetic/
	@# Rendered in a scratch copy: a no-freeze render would otherwise clear _freeze/,
	@# which holds the real-data results.
	rm -rf $(SYNTH_TMP) && mkdir -p $(SYNTH_TMP)
	tar --exclude=./_freeze --exclude=./_book --exclude=./_book_synthetic --exclude=./data \
	    --exclude=./.git --exclude=./.snakemake -cf - . | tar -xf - -C $(SYNTH_TMP)
	cd $(SYNTH_TMP) && quarto render --profile synthetic
	rm -rf _book_synthetic && cp -r $(SYNTH_TMP)/_book_synthetic . && rm -rf $(SYNTH_TMP)

render:              ## Render the book (real data; results are cached in _freeze/)
	quarto render

synthetic: test pipeline-synthetic render-synthetic   ## Everything, end to end, on synthetic data

real: test pipeline-real render                       ## Everything, end to end, on the real data

dag:                 ## Draw the pipeline graph (needs graphviz)
	JSEMICRO_DATA=synthetic $(SNAKE) --dag | dot -Tsvg > docs/pipeline-dag.svg

clean:               ## Remove rendered output (keeps data and _freeze/)
	rm -rf _book _book_synthetic .quarto

docker-build:        ## Build the image
	docker build $(PLATFORM) -t $(IMAGE) .

docker-synthetic:    ## `make synthetic` inside the container
	docker run $(PLATFORM) --rm -v "$(CURDIR)":/project $(IMAGE) make synthetic

docker-real:         ## `make real` inside the container
	docker run $(PLATFORM) --rm -v "$(CURDIR)":/project $(IMAGE) make real

docker-shell:        ## Interactive shell in the container
	docker run $(PLATFORM) --rm -it -v "$(CURDIR)":/project $(IMAGE) bash
