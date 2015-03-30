SOURCES=$(wildcard src/*.cpp)
HEADERS=$(wildcard src/*.h)

all: bin/suslix

bin/suslix: $(SOURCES) $(HEADERS)
	c++ -g -O3 -std=c++11 -DNDEBUG -mavx2 -mfma -ffast-math $(SOURCES) -lsfml-window -lsfml-graphics -lsfml-system -o bin/suslix

run: bin/suslix
	cd bin && ./suslix

profile: bin/suslix
	cd bin && ./suslix profile

.PHONY: all run profile