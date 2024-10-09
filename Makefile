##
# Rede neural artificial - Perceptron
#
# @file
# @version 0.1
CC=clang
CFLAGS=-Wall
CLIBS=-lm
TARGET=rna
# Default target
all: $(TARGET)

# Rule to build the target
$(TARGET): main.o
	$(CC) $(CFLAGS) $(CLIBS) -o $(TARGET) main.o

# Rule to compile main.c into main.o
main.o:
	$(CC) $(CFLAGS)  -c src/rna.c -o $@

# Clean up object files and the program
clean:
	rm -f *.o $(TARGET)

# Phony targets
.PHONY: all clean end

##
