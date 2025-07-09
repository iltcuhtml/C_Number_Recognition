CC = gcc
CFLAGS = -Wall -Werror -Wextra -O3 -Iinclude

SRCS = \
  src/CNN.c \
  src/image_loader.c

OBJS = $(SRCS:.c=.o)

all: train predict

train: src/train.c $(SRCS)
	$(CC) $(CFLAGS) -DNN_IMPLEMENTATION -o train src/train.c $(SRCS)

predict: src/main.c $(SRCS)
	$(CC) $(CFLAGS) -DNN_IMPLEMENTATION -o predict src/main.c $(SRCS)

clean:
	rm -f train predict $(OBJS)