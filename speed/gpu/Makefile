default: html-docs

html-docs: report.html 

%.html: %.qmd
	quarto render "$*.qmd" --to html


%.py: %.qmd
	quarto convert "$*.qmd"
	jupyter nbconvert --to python "$*.ipynb"
	rm "$*.ipynb"

%.html: %.md
	Rscript --vanilla -e "rmarkdown::render(\"$*.md\",output_format=\"html_document\")"

%.R: %.Rmd
	Rscript --vanilla -e "library(knitr); purl(\"$*.Rmd\",output=\"$*.R\")"

clean:
	$(RM) *.o *.so *.log *.aux *.out *.nav *.snm *.toc *.bak
	$(RM) Rplots.ps Rplots.pdf

fresh: clean
	$(RM) -r cache figure

