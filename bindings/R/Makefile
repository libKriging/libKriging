
ifeq ("${MAKE_SHARED_LIBS}","")
	MAKE_SHARED_LIBS=$(error MAKE_SHARED_LIBS variable must be set")
endif
ifeq ("${LIBKRIGING_PATH}","")
	LIBKRIGING_PATH=$(error LIBKRIGING_PATH variable must be set")
endif

all: build

build: ./Rlibs/.prepared
	## build package before build
	# R CMD build testBinding

	## build package after build
	mkdir -p ./Rlibs
	## --no-multiarch is only usefull for windows with visual studio mode
	export R_LIBS=$(PWD)/Rlibs && R CMD INSTALL --no-multiarch -l ./Rlibs --build testBinding
	uname -a > ./Rlibs/testBinding/.installed
	date >> ./Rlibs/testBinding/.installed

prepare: ./Rlibs/.prepared
./Rlibs/.prepared:
	mkdir -p ./Rlibs
	export R_LIBS=$(PWD)/Rlibs && Rscript testBinding/install/install.R
	uname -a > ./Rlibs/.prepared
	date >> ./Rlibs/.prepared

test: ./Rlibs/testBinding/.installed
	## output in tty
	export R_LIBS=$(PWD)/Rlibs && Rscript testBinding/test/test.R
	## output in .Rout file
	export R_LIBS=$(PWD)/Rlibs && R CMD BATCH testBinding/test/test.R

clean:
	rm -fr testBinding/src/testBinding.*
	rm -fr testBinding/src/*.o

uninstall:
	export R_LIBS=$(PWD)/Rlibs && R CMD REMOVE testBinding

veryclean: clean
	rm -fr ./Rlibs

.phony: all build prepare test clean uninstall veryclean