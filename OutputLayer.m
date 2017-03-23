function [u] = OutputLayer(x,weights)
%function [u] = HiddenLayer(tp,weights)
%This function is the output layer that computes an output u from the input
%hidden layer
%Inputs:
%x - input values of connections
%weights - weights of the inputs to the layer
%Output
%u - output of the network

u = weights*x;