#ifndef CIRCUIT_FINDER_H
#define CIRCUIT_FINDER_H

#include <algorithm>
#include <iostream>
#include <list>
#include <vector>

typedef std::list<int> NodeList;

class CircuitFinder
{
	std::vector<NodeList> AK;
	std::vector<int> Stack;
	std::vector<std::pair<int,int>> Cycles; // source/sink pairs
	std::vector<bool> Blocked;
	std::vector<NodeList> B;
	int S;
	int N;

	void unblock(int U);
	bool circuit(int V);
	void output();

public:
	CircuitFinder(std::list<int>* adj, int n_nodes){
		Blocked.resize(n_nodes);
		B.resize(n_nodes);
		AK.resize(n_nodes);
		N = n_nodes;
		for (int I = 0; I < n_nodes; ++I) {
			for (auto J = adj[I].begin(), F = adj[I].end(); J != F; ++J) {
				AK[I].push_back(*J);
			}
		}
	}

	void run();
	std::vector<std::pair<int, int>> getCycles() { return Cycles; }
};

void CircuitFinder::unblock(int U)
{
	Blocked[U - 1] = false;

	while (!B[U - 1].empty()) {
		int W = B[U - 1].front();
		B[U - 1].pop_front();

		if (Blocked[W - 1]) {
			unblock(W);
		}
	}
}

bool CircuitFinder::circuit(int V)
{
	bool F = false;
	Stack.push_back(V);
	Blocked[V - 1] = true;

	for (int W : AK[V - 1]) {
		if (W == S) {
			output();
			F = true;
		}

		// [PR request]
		//else if (W > S && !Blocked[W - 1]) {
		//	F = circuit(W);
		//}
		else if (!Blocked[W - 1]) {
			if (circuit(W))
				F = true;
		}
	}

	if (F) {
		unblock(V);
	}
	else {
		for (int W : AK[V - 1]) {
			auto IT = std::find(B[W - 1].begin(), B[W - 1].end(), V);
			if (IT == B[W - 1].end()) {
				B[W - 1].push_back(V);
			}
		}
	}

	Stack.pop_back();
	return F;
}

void CircuitFinder::output()
{
	std::cout << "circuit: ";
	for (auto I = Stack.begin(), E = Stack.end(); I != E; ++I) {
		std::cout << *I << " -> ";
	}
	std::cout << *Stack.begin() << std::endl;

	auto I = Stack.end();
	--I;
	Cycles.push_back(std::make_pair(*I, *Stack.begin()));
}

void CircuitFinder::run()
{
	Stack.clear();
	S = 1;

	while (S < N) {
		for (int I = S; I <= N; ++I) {
			Blocked[I - 1] = false;
			B[I - 1].clear();
		}
		circuit(S);

		// [PR request]
		// remove this vertex from the graph
		for (int I = S + 1; I <= N; ++I)
			AK[I - 1].remove(S);

		++S;
	}
}

#endif // CIRCUIT_FINDER_H