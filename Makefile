#Project settings
PROJECT = uneural
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
CC_FLAGS  = -O2 $(INC_FLAGS) -Wall -c -std=c99 -DFIXMATH_SATURATED_ONLY
AS_FLAGS  = $(CC_FLAGS) -D_ASSEMBLER_
LD_FLAGS = -Wall

ifeq ($(MAKECMDGOALS),test)
CC_FLAGS += -ftest-coverage -fprofile-arcs
TEST_CC_FLAGS = $(INC_FLAGS) -Wall -O2 -ftest-coverage -fprofile-arcs
TEST_LD_FLAGS += -lcmocka -l$(PROJECT) -L.
TEST_SRC = $(wildcard test/*.c)
TEST_EXEC = $(patsubst %.c, , $(TEST_SRC))
endif

ifeq ($(MAKECMDGOALS),example)
CC_FLAGS += -ggdb
EXAMPLE_CC_FLAGS = $(INC_FLAGS) -Wall -O2 -ggdb
EXAMPLE_LD_FLAGS += -lcmocka -l$(PROJECT) -L.
EXAMPLE_SRC = $(wildcard example/*.c)
EXAMPLE_EXEC = $(patsubst %.c, , $(EXAMPLE_SRC))
endif

# Find all source files
SRC_CPP = $(foreach dir, $(SRC), $(wildcard $(dir)/*.cpp))
SRC_C   = $(foreach dir, $(SRC), $(wildcard $(dir)/*.c))
SRC_S   = $(foreach dir, $(SRC), $(wildcard $(dir)/*.S))

OBJ_CPP = $(patsubst %.cpp, %.o, $(SRC_CPP))
OBJ_C   = $(patsubst %.c, %.o, $(SRC_C))
OBJ_S   = $(patsubst %.S, %.o, $(SRC_S))
OBJ     = $(OBJ_CPP) $(OBJ_C) $(OBJ_S)

# Be silent per default, but 'make V=1' will show all compiler calls.
ifneq ($(V),1)
  Q := @
  # Do not print "Entering directory ...".
  MAKEFLAGS += --no-print-directory
  # Redirect stdout/stderr for chatty tools
  NOOUT = 1> /dev/null 2> /dev/null
endif

# Compile rules.
.PHONY : all
all: lib$(PROJECT).a

test: clean lib$(PROJECT).a
	@( $(foreach T, $(TEST_SRC), $(CC) $(TEST_CC_FLAGS) $(TEST_LD_FLAGS) $(T); ./a.out;) ) 
	$(Q)rm a.out

example: clean lib$(PROJECT).a
	@( $(foreach E, $(EXAMPLE_SRC), $(CC) $(EXAMPLE_CC_FLAGS) $(EXAMPLE_LD_FLAGS) $(E);) ) 

lib$(PROJECT).a: $(OBJ)
	$(AR) rcs lib$(PROJECT).a $(OBJ)
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
	rm -f lib$(PROJECT).a $(OBJ)
