all: build

build:
	# build package before build
	# R CMD build testBinding

	# build package after build
	R CMD INSTALL --build testBinding

remove:
	R CMD REMOVE testBinding

test:
	# output in tty
	Rscript testBinding/test/test.R
	# output in .Rout file
	R CMD BATCH testBinding/test/test.R