.SUFFIXES:
MAKEFLAGS+=-r

BUILD=build

SOURCES=$(wildcard src/*.cpp src/base/*.cpp)
OBJECTS=$(SOURCES:%=$(BUILD)/%.o)

EXECUTABLE=$(BUILD)/phyx

CXXFLAGS=-g -Wall -std=c++11 -O3 -DNDEBUG -ffast-math -Isrc/microprofile

ifeq ($(shell uname),Darwin)
CXXFLAGS+=-mavx2 -mfma
LDFLAGS=-lglfw3 -framework OpenGL
else
CPUINFO=$(shell cat /proc/cpuinfo)
ifneq ($(findstring avx2,$(CPUINFO)),)
CXXFLAGS+=-mavx2
endif
ifneq ($(findstring fma,$(CPUINFO)),)
CXXFLAGS+=-mfma
endif
LDFLAGS=-lglfw -lGL -lpthread
endif

all: $(EXECUTABLE)
	./$(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(OBJECTS) $(LDFLAGS) -o $@

$(BUILD)/%.o: %
	@mkdir -p $(dir $@)
	$(CXX) $< $(CXXFLAGS) -c -MMD -MP -o $@

-include $(OBJECTS:.o=.d)
clean:
	rm -rf $(BUILD)

.PHONY: all clean
