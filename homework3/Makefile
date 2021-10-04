FILENAME := hw3
loop:
	while inotifywait -e close_write $(FILENAME).tex ; do pdflatex -shell-escape $(FILENAME).tex; pdflatex -shell-escape $(FILENAME).tex; bibtex $(FILENAME); makeglossaries $(FILENAME); pdflatex-shell-escape $(FILENAME).tex; done	


once:
	pdflatex $(FILENAME).tex; pdflatex $(FILENAME).tex; bibtex $(FILENAME); makeglossaries $(FILENAME); pdflatex $(FILENAME).tex
