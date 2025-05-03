CC = gcc
PP = g++

C_SOURCE = src
H_SOURCE = src
TEST_SOURCE = test

OBJECT = obj
EXECUTABLE = exe

CFLAGS = -Wall -Werror -Wextra -I$(H_SOURCE) -static

$(OBJECT):
	mkdir $(OBJECT)

$(EXECUTABLE):
	mkdir $(EXECUTABLE)

#make main
main: $(OBJECT)/main.o | $(EXECUTABLE)/main

#make gates
gates: $(OBJECT)/gates.o | $(EXECUTABLE)/gates

# main
$(EXECUTABLE)/main: $(OBJECT)/main.o | $(EXECUTABLE)
	$(CC) $(CFLAGS) $(OBJECT)/main.o -o $(EXECUTABLE)/main
	strip $(EXECUTABLE)/main.exe

# gates
$(EXECUTABLE)/gates: $(OBJECT)/gates.o | $(EXECUTABLE)
	$(CC) $(CFLAGS) $(OBJECT)/gates.o -o $(EXECUTABLE)/gates
	strip $(EXECUTABLE)/gates.exe

# main.c
$(OBJECT)/main.o: $(C_SOURCE)/main.c | $(OBJECT)
	$(CC) $(CFLAGS) -c $(C_SOURCE)/main.c -o $(OBJECT)/main.o

# gates.c
$(OBJECT)/gates.o: $(C_SOURCE)/gates.c | $(OBJECT)
	$(CC) $(CFLAGS) -c $(C_SOURCE)/gates.c -o $(OBJECT)/gates.o

obj_clean:
	del /Q obj\*

out_clean:
	del /Q out