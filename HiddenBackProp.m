function [delta] = HiddenBackProp(delta,weights,u)
%function [u] = HiddenBackProp(x,weights)
%This function is a hidden layer that computes an output u of the layer
%neurons
%Inputs:
%delta - delta of incoming backpropagation layer
%weights - weights of the inputs to the layer
%Output
%u - output of the current layer

sh = weights*delta;
delta = sh'.*(u.*(1-u));