.SUFFIXES:
MAKEFLAGS+=-r

BUILD=build

SOURCES=$(wildcard src/*.cpp) src/base/microprofile.cpp
OBJECTS=$(SOURCES:%=$(BUILD)/%.o)

EXECUTABLE=$(BUILD)/phyx

CXXFLAGS=-g -Wall -std=c++11 -O3 -DNDEBUG -mavx2 -mfma -ffast-math -Isrc/microprofile
LDFLAGS=-lglfw3 -framework OpenGL

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