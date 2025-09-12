# Compiler to use
CC = gcc

# Compiler warning & standard flags:
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors

# Libraries to link at the end of the link command
LDLIBS = -lm

# Default target: build the 'symnmf' executable
all: symnmf

# Link step: produce the 'symnmf' binary from its object file
symnmf: symnmf.o
	$(CC) -o symnmf symnmf.o $(LDLIBS)

# Compile step: build the object file from the C source
symnmf.o: symnmf.c
	$(CC) -c symnmf.c $(CFLAGS)

# Cleanup: remove the binary and object file
clean:
	rm -f symnmf symnmf.o