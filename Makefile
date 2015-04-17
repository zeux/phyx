SOURCES=$(wildcard src/*.cpp)
HEADERS=$(wildcard src/*.h)

all: bin/phyx

bin/phyx: $(SOURCES) $(HEADERS)
	c++ -g -O3 -std=c++11 -DNDEBUG -mavx2 -mfma -ffast-math $(SOURCES) -lsfml-window -lsfml-graphics -lsfml-system -o bin/phyx

run: bin/phyx
	cd bin && ./phyx

profile: bin/phyx
	cd bin && ./phyx profile

.PHONY: all run profile