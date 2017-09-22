#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <iostream>

#include "neuron.hpp"

class network {
public:
    network(std::vector<int> dimLayers);
    network(std::vector<int> dimLayers, float (*threshold)(float), float (*dThreshold)(float));
    network(std::string path);
    ~network();

    std::vector<std::vector<neuron>>& getNetwork();

    bool load(std::string path);
    bool save(std::string path);

	std::vector<std::vector<float>> compute(std::vector<std::vector<float>> dataset);
    std::vector<std::vector<float>> learn(std::vector<std::vector<float>> dataset, std::vector<std::vector<float>> wv, float lr = 0.1, float momentum = 0.9);
	unsigned int optimize(float threshold = 0.5);

	float getError() {
		float error = 0.f;

		for(unsigned int i = 0; i < _n[_n.size() - 1].size(); i++)
			error += fabs(_n[_n.size() - 1][i].getError());

		return error / _n[_n.size() - 1].size();
	}

private:
	std::vector<std::vector<neuron>> _n;
};

#endif // NETWORK_HPP
