#!/bin/bash
Rscript -e "rmarkdown::render('bench-full.Rmd', output_format = 'pdf_document',output_file = 'bench-full.pdf', encoding='UTF-8')"