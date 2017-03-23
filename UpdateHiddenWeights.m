function [newWeights] = UpdateHiddenWeights(weights,alpha,delta,u,momenta,lastWeight)
%function [u] = UpdateHiddenWeights(weights,alpha,delta,u,momenta,lastWeight)
%This function updates the weights after passing through a hidden layer
%Inputs:
%weights - current weights of the hidden layer
%alpha - learning rate
%u - 
%momenta - momentum
%lastWeight - last weights
%Output
%newWeights - the new weights of the hidden layer

newWeights = weights + alpha*delta*u' + momenta*lastWeight;