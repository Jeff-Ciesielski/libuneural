#Project settings
PROJECT = libuneural
LIB = 
SRC = src libfixmath/libfixmath
INC = src libfixmath/libfixmath

CROSS ?= 

#Compiler settings
CPP = $(CROSS)gcc
CC = $(CROSS)gcc
AS = $(CROSS)gcc
LD = $(CROSS)gcc
AR = $(CROSS)ar
SIZE= $(CROSS)size

INC_FLAGS = $(foreach dir, $(INC), -I$(dir))

CPP_FLAGS = -O2 $(INC_FLAGS) -Wall -c 
CC_FLAGS  = -O2 $(INC_FLAGS) -Wall -c -std=c99
AS_FLAGS  = $(CC_FLAGS) -D_ASSEMBLER_
LD_FLAGS = -Wall

# Find all source files
SRC_CPP = $(foreach dir, $(SRC), $(wildcard $(dir)/*.cpp))
SRC_C   = $(foreach dir, $(SRC), $(wildcard $(dir)/*.c))
SRC_S   = $(foreach dir, $(SRC), $(wildcard $(dir)/*.S))

OBJ_CPP = $(patsubst %.cpp, %.o, $(SRC_CPP))
OBJ_C   = $(patsubst %.c, %.o, $(SRC_C))
OBJ_S   = $(patsubst %.S, %.o, $(SRC_S))
OBJ     = $(OBJ_CPP) $(OBJ_C) $(OBJ_S)

# Compile rules.
.PHONY : all
all: $(PROJECT).a

$(PROJECT).a: $(OBJ)
	$(AR) rcs $(PROJECT).a $(OBJ)
	$(SIZE) $@

$(OBJ_CPP) : %.o : %.cpp
	$(CPP) $(CPP_FLAGS) -o $@ $<

$(OBJ_C) : %.o : %.c
	$(CC) $(CC_FLAGS) -o $@ $<

$(OBJ_S) : %.o : %.S
	$(AS) $(AS_FLAGS) -o $@ $<

# Clean rules
.PHONY : clean
clean:
	rm -f $(PROJECT).a $(OBJ)
