CC ?= gcc
CFLAGS ?= -O3 -march=native
LDLIBS = -lm
BIN = train eval record test_ntuple
all: $(BIN)
%: %.c common.h ntuple.h
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)
test: test_ntuple
	./test_ntuple
.PHONY: all test
