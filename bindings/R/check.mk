# MAKE_SHARED_LIBS and LIBKRIGING_PATH can be set
# * using shell : export VAR=VALUE
# * Using R     : Sys.setenv("VAR"="VALUE")
.check:
ifeq ("${MAKE_SHARED_LIBS}","")
	$(info empty MAKE_SHARED_LIBS env variable implies default behavior [on])
else ifeq ("${MAKE_SHARED_LIBS}","on")  # legal value
else ifeq ("${MAKE_SHARED_LIBS}","off") # legal value
else
	$(error MAKE_SHARED_LIBS env variable must be set to on or off (not '${MAKE_SHARED_LIBS}'))
endif
ifeq ("${LIBKRIGING_PATH}","")
	$(error LIBKRIGING_PATH env variable must be set using shell `export` or R `Sys.setenv`)
endif
ifeq ("$(OS)","Windows_NT")
ifeq ("${EXTRA_SYSTEM_LIBRARY_PATH}","")
	$(error EXTRA_SYSTEM_LIBRARY_PATH env variable must be set using shell `export` or R `Sys.setenv`)
endif
endif
