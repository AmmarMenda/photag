CC = gcc
CFLAGS = -Wall -O2
INCLUDES = -I./include
# Tell the linker to look in the ./lib folder for libraries
LIB_PATHS = -L./lib
LIBS = $(LIB_PATHS) -Wl,-rpath,'$$ORIGIN/lib' -lglfw -lGLEW -lGL -lm -ldarknet -lpthread

TARGET = gallery
SRC = src/main.c
OBJ = $(SRC:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $(TARGET) $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)
