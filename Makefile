CC = gcc
CFLAGS = -Wall -Werror -Wextra -O3 -Iinclude -I"C:/msys64/ucrt64/include/SDL3"
LDFLAGS = -L"C:/msys64/ucrt64/lib" -lSDL3

all: train predict

train: $(OBJS)
	$(CC) $(CFLAGS) -o train src/train.c $(LDFLAGS)

predict: $(OBJS)
	$(CC) $(CFLAGS) -o predict src/main.c $(LDFLAGS)