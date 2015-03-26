all: bin/suslix

bin/suslix:
	c++ -g -O3 -std=c++11 -DNDEBUG -mavx2 -ffast-math src/main.cpp -lsfml-window -lsfml-graphics -lsfml-system -o bin/suslix

run: bin/suslix
	cd bin && ./suslix

profile: bin/suslix
	cd bin && ./suslix profile

.PHONY: all run profile
.PHONY: bin/suslix