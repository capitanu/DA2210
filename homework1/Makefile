all:
	while inotifywait -e close_write *.tex ; do pdflatex *.tex; done
