function [u] = HiddenLayer(x,weights)
%function [u] = HiddenLayer(tp,weights)
%This function is a hidden layer that computes an output u of the layer
%neurons
%Inputs:
%x - input values of connections
%weights - weights of the inputs to the layer
%Output
%u - output of the current layer

sigma = weights*x';
u = logsig(sigma);