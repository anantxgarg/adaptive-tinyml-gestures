CC ?= gcc
CFLAGS ?= -std=c99 -O2 -Wall -Wextra -Werror

device_sim: main.c tinyml.c features.c tinyml.h features.h model_data.h
	$(CC) $(CFLAGS) -o $@ main.c tinyml.c features.c -lm

clean:
	rm -f device_sim

.PHONY: clean
