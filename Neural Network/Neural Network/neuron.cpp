#include <algorithm>

#include "neuron.hpp"

// Fonctions d'activation avec leur dérivées
float ftanh(float x) { return static_cast<float>(tanh(static_cast<float>(x))); }
float dftanh(float x) { return static_cast<float>(1.0-pow(ftanh(x), 2)); }

float sigmoid(float x) { return static_cast<float>(1.0/(1.0+exp(-x))); }
float dsigmoid(float x) { return exp(-x)/pow(1+exp(-x), 2); }
////////////////////////////////////////////


neuron::neuron(float out) : _out(out) {
    setFunction();
    setDerivatedFunction();
}

neuron::neuron(std::vector<neuron*> backVector, float out) : _out(out) {
    link(backVector);

    setFunction();
    setDerivatedFunction();
}

neuron::neuron(std::vector<neuron*> backVector, std::vector<float> weightsVector, float out) : _out(out) {
    link(backVector, weightsVector);

    setFunction();
    setDerivatedFunction();
}

 neuron::neuron(float (*threshold)(float), float (*dThreshold)(float)) {
     setFunction(threshold);
    setDerivatedFunction(dThreshold);
 }

neuron::~neuron() {
    // Détruire links
}


void neuron::operator=(float out) {
    _out = out;
}


float neuron::getOut() {
    return _out;
}

std::vector<neuron*> neuron::getBackNeurons() {
    return _backNeurons;
}

std::vector<float> neuron::getWeights() {
    return _weights;
}


void neuron::setWeight(int weightId, float value) {
    _weights[weightId] = value;
}


void neuron::link(neuron &n, float weight) {
    _backNeurons.push_back(&n);
    _weights.push_back(weight);
    _deltaWeights.push_back(0.0);
}

void neuron::link(std::vector<neuron*> backVector) {
    _backNeurons.insert(_backNeurons.end(), backVector.begin(), backVector.end());
    _weights.resize(_backNeurons.size(), 0.0);
    _deltaWeights.resize(_backNeurons.size(), 0.0);
}

void neuron::link(std::vector<neuron*> backVector, std::vector<float> weightsVector) {
    _backNeurons.insert(_backNeurons.end(), backVector.begin(), backVector.end());
    _weights.insert(_weights.end(), weightsVector.begin(), weightsVector.end());
    _deltaWeights.resize(_backNeurons.size(), 0.0);
}

void neuron::unlink(int index) {
    _weights.erase(_weights.begin() + index);
    _deltaWeights.erase(_deltaWeights.begin() + index);
    _backNeurons.erase(_backNeurons.begin() + index);
}


void neuron::setFunction(std::function<float(float)> threshold) {
    _threshold = threshold;
}

void neuron::setDerivatedFunction(std::function<float(float)> dThreshold) {
    _dThreshold = dThreshold;
}

float neuron::compute() {
    float sum = 0.f;

    if (_backNeurons.size() == 0)
        return _out;

    for(unsigned int i = 0; i < _backNeurons.size(); i++)
        sum += _backNeurons[i]->compute() * _weights[i];

    _out = _threshold(sum);

    return _out;
}

float neuron::learn(float wv, float lr, float momentum) {
    float sum = 0.f;

    if (_backNeurons.size() == 0)
        return _out;

    for(unsigned int i = 0; i < _backNeurons.size(); i++)
        sum += _backNeurons[i]->compute() * _weights[i];

    _out = _threshold(sum);

    _error = wv - _out;

    float dW;
    for(unsigned int i = 0; i < _weights.size(); i++) {
        dW = _error * _backNeurons[i]->backprop(_dThreshold(_error*_weights[i]), lr) * lr + momentum * _deltaWeights[i];
        _weights[i] += dW;
        _deltaWeights[i] = dW;
    }

    return _out;
}

float neuron::backprop(float error, float lr, float momentum) {
    if (_backNeurons.size() == 0)
        return _out;

    float dW;
    for(unsigned int i = 0; i < _weights.size(); i++) {
        dW = error * _backNeurons[i]->backprop(_dThreshold(error*_weights[i])) * lr + momentum * _deltaWeights[i];
        _weights[i] += dW;
        _deltaWeights[i] = dW;
    }

    return _out;
}
