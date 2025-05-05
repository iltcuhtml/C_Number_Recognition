CC = gcc

C_SOURCE = src
H_SOURCE = src

OBJECT = obj
EXECUTABLE = exe

INCLUDE_PATH = C:/raylib/w64devkit/x86_64-w64-mingw32/include
LIBRARY_PATH = C:/raylib/w64devkit/x86_64-w64-mingw32/lib

CFLAGS = -Wall -Werror -Wextra -O3 -I$(INCLUDE_PATH)
LDFLAGS = -lraylib -static-libgcc -static-libstdc++

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

obj_clean:
	del /Q $(OBJECT)\*

exe_clean:
	del /Q $(EXECUTABLE)\*

clean: obj_clean exe_clean