
all: .check build

include ./check.mk

build: ./Rlibs/.prepared
	## build package before build
	# R CMD build rlibkriging

	## build package after build
	mkdir -p ./Rlibs
	## --no-multiarch is only usefull for windows with visual studio mode
	export R_LIBS=$(PWD)/Rlibs && Rscript -e "Rcpp::compileAttributes(pkgdir = 'rlibkriging', verbose = TRUE)"
	#export R_LIBS=$(PWD)/Rlibs && Rscript -e "roxygen2::roxygenise(package.dir = 'rlibkriging')"
	export R_LIBS=$(PWD)/Rlibs && R CMD INSTALL --no-multiarch -l ./Rlibs --build rlibkriging
	uname -a > ./Rlibs/rlibkriging/.installed
	date >> ./Rlibs/rlibkriging/.installed

bench: ./Rlibs/.prepared
  export R_LIBS=$(PWD)/Rlibs && Rscript -e "rmarkdown::render('bench-full.Rmd', output_format = 'pdf_document',output_file = 'bench-full.pdf', encoding='UTF-8')"

prepare: ./Rlibs/.prepared
./Rlibs/.prepared:
	mkdir -p ./Rlibs
	export R_LIBS=$(PWD)/Rlibs && Rscript rlibkriging/install/install.R
	uname -a > ./Rlibs/.prepared
	date >> ./Rlibs/.prepared

test: ./Rlibs/rlibkriging/.installed
	## output in tty
	export R_LIBS=$(PWD)/Rlibs && cd rlibkriging/tests && Rscript testthat.R
#	## output in testthat.Rout file
#	export R_LIBS=$(PWD)/Rlibs && cd rlibkriging/tests && R CMD BATCH testthat.R

clean:
	rm -fr rlibkriging/src/rlibkriging.*
	rm -fr rlibkriging/src/*.o

uninstall:
	export R_LIBS=$(PWD)/Rlibs && R CMD REMOVE rlibkriging

veryclean: clean
	rm -fr ./Rlibs

.phony: all build prepare test clean uninstall veryclean
