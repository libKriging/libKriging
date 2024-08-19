#!/bin/bash
Rscript -e "rmarkdown::render('bench_perf-full.Rmd', output_format = 'pdf_document',output_file = 'bench_perf-full.pdf', encoding='UTF-8')"