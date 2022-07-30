######################################################################
# @author      : juhua (juhua@juhua-Z690-UD-AX-DDR4)
# @file        : makefile
# @created     : 週三  7月 27, 2022 11:51:38 CST
######################################################################

# This flag is to control to use release or debug mode to compile code files.
debug ?= 0


.PHONY: clean all

# Compile the target file with the object files.
all: 
	cd ./tensorEye; $(MAKE) debug=$(debug)
	cd ./simpleNet; $(MAKE) debug=$(debug)
	cd ./dcgan_strurct; $(MAKE) debug=$(debug)


# Clean all compiled files.
clean:
	cd ./tensorEye; $(MAKE) clean
	cd ./simpleNet; $(MAKE) clean
	cd ./dcgan_strurct; $(MAKE) clean


