CC = gcc

C_SOURCE = src
H_SOURCE = src

OBJECT = obj
EXECUTABLE = exe

INCLUDE_PATH = C:/raylib/w64devkit/x86_64-w64-mingw32/include
LIBRARY_PATH = C:/raylib/w64devkit/x86_64-w64-mingw32/lib

CFLAGS = -Wall -Werror -Wextra -O3 -I$(INCLUDE_PATH)
LDFLAGS = -L$(LIBRARY_PATH) -lraylib

$(OBJECT):
	mkdir $(OBJECT)

$(EXECUTABLE):
	mkdir $(EXECUTABLE)

main: $(EXECUTABLE)/main.exe

$(EXECUTABLE)/main.exe: $(OBJECT)/main.o | $(EXECUTABLE)
	$(CC) $(OBJECT)/main.o -o $(EXECUTABLE)/main.exe $(LDFLAGS)
	strip $(EXECUTABLE)/main.exe

$(OBJECT)/main.o: $(C_SOURCE)/main.c | $(OBJECT)
	$(CC) $(CFLAGS) -c $(C_SOURCE)/main.c -o $(OBJECT)/main.o

gym: $(EXECUTABLE)/gym.exe

$(EXECUTABLE)/gym.exe: $(OBJECT)/gym.o | $(EXECUTABLE)
	$(CC) $(OBJECT)/gym.o -o $(EXECUTABLE)/gym.exe $(LDFLAGS)
	strip $(EXECUTABLE)/gym.exe

$(OBJECT)/gym.o: $(C_SOURCE)/gym.c | $(OBJECT)
	$(CC) $(CFLAGS) -c $(C_SOURCE)/gym.c -o $(OBJECT)/gym.o

adder: $(EXECUTABLE)/adder.exe

$(EXECUTABLE)/adder.exe: $(OBJECT)/adder.o | $(EXECUTABLE)
	$(CC) $(OBJECT)/adder.o -o $(EXECUTABLE)/adder.exe $(LDFLAGS)
	strip $(EXECUTABLE)/adder.exe

$(OBJECT)/adder.o: $(C_SOURCE)/adder.c | $(OBJECT)
	$(CC) $(CFLAGS) -c $(C_SOURCE)/adder.c -o $(OBJECT)/adder.o

adder_gen: $(EXECUTABLE)/adder_gen.exe

$(EXECUTABLE)/adder_gen.exe: $(OBJECT)/adder_gen.o | $(EXECUTABLE)
	$(CC) $(OBJECT)/adder_gen.o -o $(EXECUTABLE)/adder_gen.exe $(LDFLAGS)
	strip $(EXECUTABLE)/adder_gen.exe

$(OBJECT)/adder_gen.o: $(C_SOURCE)/adder_gen.c | $(OBJECT)
	$(CC) $(CFLAGS) -c $(C_SOURCE)/adder_gen.c -o $(OBJECT)/adder_gen.o

obj_clean:
	del /Q $(OBJECT)\*

exe_clean:
	del /Q $(EXECUTABLE)\*

clean: obj_clean exe_clean