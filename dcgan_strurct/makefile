######################################################################
# @author      : juhua (juhua@juhua-Z690-UD-AX-DDR4)
# @file        : makefile
# @created     : 週三  7月 27, 2022 11:47:27 CST
######################################################################

# This flag is to control to use release or debug mode to compile code files.
debug ?= 0

# These directories is set to compile this project.
Includes_Dir = 
Sourc_Dir = ./src
Bin_Dir = ../bin/dcgan_struct

# Set the code file extension.
File_Extension := cpp

# Set compiler and flags.
CC = g++
CFLAGS = $(shell pkg-config --cflags libtorch)
LIBS =  $(shell pkg-config --libs libtorch) -lpthread

# List all source code in the src directory
Sourcs := $(wildcard $(Sourc_Dir)/*.$(File_Extension))

# Create the list to generate object files
Objects := $(patsubst $(Sourc_Dir)/%.$(File_Extension), $(Bin_Dir)/%.o, $(Sourcs))

# Set the target file name
Target = dcgan_struct

# Switch compile mode
ifeq ($(debug), 1)
	CFLAGS += -g
else
	CFLAGS += -O3
endif


.PHONY: clean all install uninstall

# Compile the target file with the object files.
all: $(Bin_Dir)/$(Target)

$(Bin_Dir)/$(Target): $(Objects)
	$(CC) $^ $(CFLAGS) -o $@ $(LIBS)

# Compile the source code to the object files.
$(Bin_Dir)/%.o: $(Sourc_Dir)/%.$(File_Extension)
	mkdir -p $(Bin_Dir)
	$(CC) -c $< $(CFLAGS) -o $@ 


# Clean all compiled files.
clean:
	rm -f $(Bin_Dir)/*


