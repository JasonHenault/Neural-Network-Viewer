#ifndef NEURON_HPP
#define NEURON_HPP

#include <vector>
#include <cmath>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/variant.hpp>

// Prototype des fonctions d'activation et de leur dérivées
float ftanh(float x);
float dftanh(float x);

float sigmoid(float x);
float dsigmoid(float x);
///////////////////////////////////////////////////////////

class neuron {
public:
    neuron(float out = 0.0);
    neuron(std::vector<neuron*> backVector, float out = 0.0);
    neuron(std::vector<neuron*> backVector, std::vector<float> weightsVector, float out = 0.0);
    neuron(float (*threshold)(float), float (*dThreshold)(float)); // Utilisé pour la classe network
    ~neuron();

    void operator=(float out);

    float getOut();
	float getError() { return _error; };
    std::vector<neuron*> getBackNeurons();
    std::vector<float> getWeights();

    void setWeight(int weightId, float value);

    void link(neuron &n, float weight = 0.0); // Ajout d'un neurone en entrée
    void link(std::vector<neuron*> backVector); // Ajout d'un vecteur de neurones en entrée
    void link(std::vector<neuron*> backVector, std::vector<float> weightsVector);// Ajout d'un vecteur de neurones en entrée et de leur poids associés
    void unlink(int index);

    void setFunction(std::function<float(float)> threshold = ftanh); // définition de la fonction de seuil
    void setDerivatedFunction(std::function<float(float)> dThreshold = dftanh); // définition de la fonction de seuil

    float compute(); // lr = learn rate
    float learn(float wv, float lr = 0.1f, float momentum = 0.9f); // wv = want value, lr = learn rate

    // Fonction de sérialisation pour enregistrement du network
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
        ar & _backNeurons;
        ar & _weights;
        ar & _deltaWeights;

		ar & _out;

		//ar & boost::variant<_threshold, _dThreshold>;
    }
    ///////////////////////////////////////////////////////////

protected:
    float backprop(float error, float lr = 0.1, float momentum = 0.0);

private:
    std::vector<neuron*> _backNeurons; // pointeur vers le neurone précédent
    std::vector<float> _weights; // poids synaptique
    std::vector<float> _deltaWeights; // variable à t-1 des poids

    float _out, _error;

	std::function<float(float)> _threshold; // pointeur vers fonction de seuil
	std::function<float(float)> _dThreshold; // pointeur vers fonction dérivé de seuil (backpropagation)
};

#endif // NEURON_HPP
