CC = gcc
PP = g++

C_SOURCE = src
H_SOURCE = src
TEST_SOURCE = test

OBJECT = obj
EXECUTABLE = exe

CFLAGS = -Wall -Werror -Wextra -O3 -I$(H_SOURCE) -static

$(OBJECT):
	mkdir $(OBJECT)

$(EXECUTABLE):
	mkdir $(EXECUTABLE)

#make main
main: $(OBJECT)/main.o | $(EXECUTABLE)/main

# main
$(EXECUTABLE)/main: $(OBJECT)/main.o | $(EXECUTABLE)
	$(CC) $(CFLAGS) $(OBJECT)/main.o -o $(EXECUTABLE)/main
	strip $(EXECUTABLE)/main.exe

# main.c
$(OBJECT)/main.o: $(C_SOURCE)/main.c | $(OBJECT)
	$(CC) $(CFLAGS) -c $(C_SOURCE)/main.c -o $(OBJECT)/main.o

obj_clean:
	del /Q obj\*

out_clean:
	del /Q out