
all: .check build

include ./check.mk

build: ./Rlibs/.prepared
	## build package before build
	# R CMD build rlibkriging

	## build package after build
	mkdir -p ./Rlibs
	## --no-multiarch is only usefull for windows with visual studio mode
	export R_LIBS=$(PWD)/Rlibs && Rscript -e "Rcpp::compileAttributes(pkgdir = 'rlibkriging', verbose = TRUE)"
	export R_LIBS=$(PWD)/Rlibs && R CMD INSTALL --no-multiarch -l ./Rlibs --build rlibkriging
	uname -a > ./Rlibs/rlibkriging/.installed
	date >> ./Rlibs/rlibkriging/.installed

prepare: ./Rlibs/.prepared
./Rlibs/.prepared:
	mkdir -p ./Rlibs
	export R_LIBS=$(PWD)/Rlibs && Rscript rlibkriging/install/install.R
	uname -a > ./Rlibs/.prepared
	date >> ./Rlibs/.prepared

test: ./Rlibs/rlibkriging/.installed
	## output in tty
	export R_LIBS=$(PWD)/Rlibs && Rscript rlibkriging/tests/testthat.R
	## output in .Rout file
	export R_LIBS=$(PWD)/Rlibs && R CMD BATCH rlibkriging/tests/testthat.R

clean:
	rm -fr rlibkriging/src/rlibkriging.*
	rm -fr rlibkriging/src/*.o

uninstall:
	export R_LIBS=$(PWD)/Rlibs && R CMD REMOVE rlibkriging

veryclean: clean
	rm -fr ./Rlibs

.phony: all build prepare test clean uninstall veryclean