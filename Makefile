BUILD=build
TEST=test
CC=g++
CFLAGS=-Wall -Wextra -g

all: main $(BUILD)/goopy.o

main: main.cpp $(BUILD)/goopy.o
	$(CC) $(CFLAGS) main.cpp $(BUILD)/goopy.o -lm -o $@

$(BUILD)/goopy.o: goopy.cpp goopy.h | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD):
	mkdir -pv $(BUILD)

leak: main
	valgrind --leak-check=full -s ./main

clean:
	rm -rf build/*
	rm -f  main
