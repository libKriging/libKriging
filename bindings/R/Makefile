all: build

build:
	# build package before build
	# R CMD build testBinding

	# build package after build
	mkdir -p ./Rlibs
	R CMD INSTALL -l ./Rlibs --build testBinding

remove:
	export R_LIBS=$(PWD)/Rlibs && R CMD REMOVE testBinding

test:
	# output in tty
	export R_LIBS=$(PWD)/Rlibs && Rscript testBinding/test/test.R
	# output in .Rout file
	export R_LIBS=$(PWD)/Rlibs && R CMD BATCH testBinding/test/test.R

clean:
	rm -fr testBinding/src/installed
	rm -fr testBinding/src/build
	rm -fr testBinding/src/testBinding.*
	rm -fr testBinding/src/test_binding.o
