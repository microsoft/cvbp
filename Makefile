########################################################################
#
# Makefile for pre-built ML model
#
# Copyright (c) Graham.Williams@togaware.com
#
# License: Creative Commons Attribution-ShareAlike 4.0 International.
#
########################################################################

# Include standard Makefile templates.

INC_BASE    = $(HOME)/.local/share/make
INC_PANDOC  = $(INC_BASE)/pandoc.mk
INC_GIT     = $(INC_BASE)/git.mk
INC_MLHUB   = $(INC_BASE)/mlhub.mk
INC_CLEAN   = $(INC_BASE)/clean.mk

ifneq ("$(wildcard $(INC_PANDOC))","")
  include $(INC_PANDOC)
endif
ifneq ("$(wildcard $(INC_GIT))","")
  include $(INC_GIT)
endif
ifneq ("$(wildcard $(INC_MLHUB))","")
  include $(INC_MLHUB)
endif
ifneq ("$(wildcard $(INC_CLEAN))","")
  include $(INC_CLEAN)
endif

clean::
	rm -rf README.txt

realclean:: clean
	rm -f $(MODEL)_*.mlm thumbnail.jpg
