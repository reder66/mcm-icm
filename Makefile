# Courtesy of https://tex.stackexchange.com/a/40759

BUILDDIR = _build
PROJECT = 82104
COMPILER = xelatex

.PHONY: all clean

all: $(PROJECT).pdf
	open $(PROJECT).pdf

$(PROJECT).pdf: mcmthesis-demo.pdf
	cp $(BUILDDIR)/mcmthesis-demo.pdf $(PROJECT).pdf

mcmthesis-demo.pdf: mcmthesis-demo.tex
	mkdir -p $(BUILDDIR)
	latexmk -pdf -pdflatex="$(COMPILER) -interaction=nonstopmode" \
		-outdir=$(BUILDDIR) -use-make mcmthesis-demo.tex

clean:
	latexmk -C -outdir=$(BUILDDIR)
	rm -f $(PROJECT).pdf

