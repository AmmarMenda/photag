CC = gcc
CFLAGS = -Wall -O2
INCLUDES = -I./include
LIB_PATHS = -L./lib
LIBS = $(LIB_PATHS) -Wl,-rpath,'$$ORIGIN/lib' -lglfw -lGLEW -lGL -lm -ldarknet -lpthread

TARGET = gallery
SRC = src/main.c
OBJ = $(SRC:.c=.o)

PREFIX ?= /usr
BINDIR = $(PREFIX)/bin
LIBDIR = $(PREFIX)/lib

.PHONY: all compile install clean

# Builds the target executable locally
all: $(TARGET)

# Compiles object files without linking
compile: $(OBJ)

$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $(TARGET) $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

install: $(TARGET)
	install -d $(DESTDIR)$(BINDIR)
	install -d $(DESTDIR)$(LIBDIR)
	install -m 755 $(TARGET) $(DESTDIR)$(BINDIR)/$(TARGET)
	install -m 755 lib/libdarknet.so $(DESTDIR)$(LIBDIR)/libdarknet.so

clean:
	rm -f $(OBJ) $(TARGET)