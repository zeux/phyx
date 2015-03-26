all:
	c++ -g -O3 -std=c++11 -DNDEBUG -mavx2 -ffast-math src/main.cpp -lsfml-window -lsfml-graphics -lsfml-system -o bin/suslix
	cd bin && ./suslix
