#include<iostream>
#include <sstream>
#include <fstream>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include "network.hpp"

using namespace std;

network::network(std::vector<int> dimLayers) {
    // Création des neurones
    _n.resize(dimLayers.size());

    for(unsigned int layer = 0; layer < dimLayers.size(); layer++)
        for(int nIndex = 0; nIndex < dimLayers[layer]; nIndex++)
            _n[layer].push_back(neuron());

    // Linkage (full)
    for(unsigned int layer = dimLayers.size() - 1; layer >= 1; layer--)
        for(int nIndex = 0; nIndex < dimLayers[layer]; nIndex++)
            for(int backLayer = layer-1; backLayer >= 0; backLayer--)
                for(int nBackIndex = 0; nBackIndex < dimLayers[backLayer]; nBackIndex++)
                    _n[layer][nIndex].link(_n[backLayer][nBackIndex]);
}

network::network(std::vector<int> dimLayers, float (*threshold)(float), float (*dThreshold)(float)) {
    // Création des neurones
    _n.resize(dimLayers.size());

    for(unsigned int layer = 0; layer < dimLayers.size(); layer++)
        for(int nIndex = 0; nIndex < dimLayers[layer]; nIndex++)
            _n[layer].push_back(neuron(threshold, dThreshold));

    // Linkage (full)
    for(unsigned int layer = dimLayers.size() - 1; layer >= 1; layer--)
        for(int nIndex = 0; nIndex < dimLayers[layer]; nIndex++)
            for(int backLayer = layer-1; backLayer >= 0; backLayer--)
                for(int nBackIndex = 0; nBackIndex < dimLayers[backLayer]; nBackIndex++)
                    _n[layer][nIndex].link(_n[backLayer][nBackIndex]);
}

network::network(std::string path) {
    load(path);
}


network::~network() {
}


std::vector<std::vector<neuron>>& network::getNetwork() {
    return _n;
}


bool network::load(std::string path) {
    ifstream ifs(path.c_str());

    if(!ifs)
        return false;

    boost::archive::text_iarchive infs(ifs);
    infs & _n;

    return true;
}

bool network::save(std::string path) {
    ofstream ofs(path);

    if(!ofs)
        return false;

    boost::archive::text_oarchive to(ofs);
    to & _n;

    return true;
}


std::vector<std::vector<float>> network::compute(std::vector<std::vector<float>> dataset) {
	std::vector<std::vector<float>> out;
	out.resize(dataset.size());

	for (unsigned int dsIndex = 0; dsIndex < dataset.size(); dsIndex++) {
		for (unsigned int nIndex = 0; nIndex < _n[0].size(); nIndex++)
			_n[0][nIndex] = dataset[dsIndex][nIndex];

		for (unsigned int nIndex = 0; nIndex < _n[_n.size() - 1].size(); nIndex++)
			out[dsIndex].push_back(_n[_n.size() - 1][nIndex].compute());
	}

    return out;
}

std::vector<std::vector<float>> network::learn(std::vector<std::vector<float>> dataset, std::vector<std::vector<float>> wv, float lr, float momentum) {
    std::vector<std::vector<float>> out;
    out.resize(dataset.size());

    for(unsigned int dsIndex = 0; dsIndex < dataset.size(); dsIndex++) {
        for(unsigned int nIndex = 0; nIndex < _n[0].size(); nIndex++)
            _n[0][nIndex] = dataset[dsIndex][nIndex];

        for(unsigned int nIndex = 0; nIndex < _n[_n.size()-1].size(); nIndex++)
            out[dsIndex].push_back(_n[_n.size()-1][nIndex].learn(wv[dsIndex][nIndex], lr, momentum));
    }

    return out;
}

unsigned int network::optimize(float threshold) {
    unsigned int nbDeleted = 0;

    for(unsigned int layer = _n.size() - 1; layer >= 1; layer--) // LeCun optimisation
        for(unsigned int nIndex = 0; nIndex < _n[layer].size(); nIndex++) {
            std::vector<float> weights = _n[layer][nIndex].getWeights();
            for(int backLayer = layer-1; backLayer >= 0; backLayer--)
                for(unsigned int nBackIndex = 0; nBackIndex < _n[backLayer].size(); nBackIndex++)
                    if (fabs(weights[nBackIndex]) < threshold) {
                        _n[layer][nIndex].unlink(nBackIndex);
                        nbDeleted++;
                    }
        }

    return nbDeleted;
}
