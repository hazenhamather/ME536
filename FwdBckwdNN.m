%% 
clear variables
close all
clc

%Initialization
tp = [0 0;1 0;0 1;1 1];
d0 = [0;1;1;0];
numInputs = 2;
numHiddenLayers = 1;
numOutputs = 1;
learningRate = [0.25;0.30];
momentum = [0.7;0.8];
deltaw1last = 0;
deltaw2last = 0;
epoch = 0;
targetError = 1e-4;
maxEpochs = 100000;
error = Inf*ones(length(tp),1);
epochError = Inf;
%Randomizing starting weights
deltaw1 = rand(3,3);
deltaw2 = rand(1,3);
alpha = learningRate(1);
alpha = 0.65;
momenta = momentum(1);
% deltaw1 = [0.1 0.7 0.4;0.9 0.67 0.83];
% deltaw2 = [0.15 0.49];

%% Enter the training loop
while epoch < maxEpochs && epochError > targetError
    epoch = epoch + 1;
    trainingIndex = randperm(4); %randomizing the training data
    for i = trainingIndex
        %Forward Propagate
%         i = 1;
        x = [tp(i,:) 1];
        uh1 = HiddenLayer(x,deltaw1);
        uo = OutputLayer(uh1,deltaw2);
        
        %Back Propagate 
        deltao = (d0(i) - uo);
        error(i) = deltao;
        deltah1 = HiddenBackProp(deltao,deltaw2,uh1);
        deltaw2 = UpdateHiddenWeights(deltaw2,alpha,deltao,uh1,momenta,deltaw2last);
        deltaw1 = UpdateInputWeights(deltaw1,alpha,deltah1,x,momenta,deltaw1last);
        
        %Setting the previous weight
        deltaw2last = alpha*deltao*uh1' + momenta*deltaw2last;
        deltaw1last = alpha*deltah1*x + momenta*deltaw1last;
    end
    totalSquaredError(epoch) = error'*error;
    epochError = totalSquaredError(epoch);
end

%% Testing Phase
for i = trainingIndex
    x = tp(i,:);
    sigmah = deltaw1*[x 1]';
    uh = logsig(sigmah);
    sigmao = deltaw2*uh;
    uo = sigmao
    error = (d0(i) - uo);
end

%% Writing to Files
% filename = 'LR25M7.xlsx';
% filename = 'LR25M8.xlsx';
% filename = 'LR30M7.xlsx';
% filename = 'LR30M8.xlsx';
% filename = 'LR55M7.xlsx';
% filename = 'LR60M7.xlsx';
% filename = 'LR65M7.xlsx';
filename = 'LR65M7-3neurons.xlsx';
xlswrite(filename,deltaw1,'Layer1');
xlswrite(filename,deltaw2,'Layer2');
xlswrite(filename,totalSquaredError','Squared Error');