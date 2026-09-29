# Makefile -- TAE @ NeurIPS 2026 workshop paper
# Primary build: latexmk. Fallback: tectonic (handles bibtex + reruns itself).
# Usage: make | make pdf | make clean | make distclean
#
# macros.tex lives in ../shared/ and is reached by an explicit ../shared/ path
# in the sources (\input), which tectonic and latexmk both resolve.
# references.bib does NOT: Overleaf's bibtex refuses a parent-directory path,
# so the bibliography sits beside main.tex and no BIBINPUTS is needed. The
# vendored neurips_2026.sty is not shared either: each venue carries its own
# copy (an unchanging upstream build dependency, and Overleaf expects the style
# inside the project). Figures/tables resolve via explicit ../figures and
# ../tables paths. See paper/README.md.

MAIN := main

.PHONY: all pdf clean distclean

all: pdf

pdf:
	@if command -v latexmk >/dev/null 2>&1; then \
		latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex; \
	elif command -v tectonic >/dev/null 2>&1; then \
		tectonic --keep-intermediates --keep-logs $(MAIN).tex; \
	else \
		echo "ERROR: no TeX toolchain found (install latexmk or tectonic," ; \
		echo "or build on Overleaf -- see README.md)"; \
		exit 1; \
	fi

clean:
	@if command -v latexmk >/dev/null 2>&1; then latexmk -c $(MAIN).tex; fi
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out \
	      $(MAIN).fls $(MAIN).fdb_latexmk $(MAIN).synctex.gz $(MAIN).toc \
	      $(MAIN).nav $(MAIN).snm $(MAIN).vrb \
	      sections/*.aux

distclean: clean
	rm -f $(MAIN).pdf
