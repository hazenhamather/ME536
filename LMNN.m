%% 
clear variables
close all
clc

%Initialization
tp = [0 0;1 0;0 1;1 1];
d0 = [0;1;1;0];
numHiddenLayers = 1;
numInputs = 2;
numOutputs = 1;
numNeurons = 3;
learningRate = [0.25;0.30];
momentum = [0.7;0.8];
deltaw1last = 0;
deltaw2last = 0;
epoch = 0;
targetError = 1e-4;
maxEpochs = 30000;
error = Inf*ones(length(tp),1);
epochError = Inf;
J = [];
mu = 0.1;
beta = 10;

%Randomizing starting weights
w1 = rand(numNeurons,numInputs + 1);
w2 = rand(numOutputs,numNeurons);
% alpha = learningRate(1);
alpha = 0.65;
% momenta = momentum(1);
% w1 = [.9027 0.9448 0.4909;0.3377 0.9001 0.3692];
% w2 = [0.4893 0.1112];
% w1 = [0.1 0.7 0.4;0.9 0.67 0.83];
%       [.4909,.3692,  0  ;
%       .9027,.3377,.4893;
%       .9448,.9001,.1112] ;

% w2 = [0.15 0.49];

%% Enter the training loop
while epoch < maxEpochs && epochError > targetError
%     disp(epochError)
    epoch = epoch + 1;
    trainingIndex = randperm(4); %randomizing the training data
    J = [];
%     for i = trainingIndex
    for i = [1 2 3 4]
        %Forward Propagate
%         i = 1;
        x = [tp(i,:) 1];
        uh1 = HiddenLayer(x,w1);
        uo = OutputLayer(uh1,w2);
        
        %Back Propagate 
        deltao = 1;
        error(i) = (d0(i) - uo);
        deltah1 = HiddenBackProp(deltao,w2,uh1);
        
        %Formation of the Jacobian
        temp = [deltah1(1)*x deltah1(2)*x deltah1(3)*x deltao*uh1'];
        J = [J; temp];
    end
    repeat = 1;
    epochError = error'*error;
    previousError = error;
    JJ = J'*J;
    [row,col] = size(JJ);
    id = eye(row,col);
    while repeat
        %Step 4
        change = ((JJ + mu*id)^(-1))*J'*previousError;
        
        %Step 5
        counter = 1;
        for i = 1:numInputs+1:numNeurons*numInputs+numNeurons
           tempdeltaw1(counter,:) = w1(counter,:) + change(i:i+2)';
           counter = counter + 1;
        end
        
%         tempdeltaw1(1,:) = w1(1,:) + change(1:3)';
%         tempdeltaw1(2,:) = w1(2,:) + change(4:6)';
%         tempdeltaw1(3,:) = w1(3,:) + change(7:9)';
        
%         counter = 1;
%         for i = 1:numNeurons
        tempdeltaw2 = w2 + change(numNeurons*numInputs+numNeurons+1:end)';
%         tempdeltaw2 = w2 + change(10:end)';
        
        % repeat fwd propagation and compute new error
        for i = [1 2 3 4]
            x = [tp(i,:) 1];
            uhtemp = HiddenLayer(x,tempdeltaw1);
            uotemp = OutputLayer(uhtemp,tempdeltaw2);
            error(i) = (d0(i) - uotemp);
        end
        currentError = error'*error;
        
        %Step 6
        if currentError < epochError
            mu = mu/beta;
            w1 = tempdeltaw1;
            w2 = tempdeltaw2;
            repeat = 0;
            epochError = currentError;
            total(epoch) = epochError;
        elseif mu > norm(J)
            w1 = tempdeltaw1;
            w2 = tempdeltaw2;
            repeat = 0;
            epochError = currentError;
            total(epoch) = epochError;
        else
            mu = mu*beta;
        end
    end
end

%% Plotting
data = xlsread('LR65M7-3neurons.xlsx','Squared Error');
loglog(1:epoch,total,1:length(data),data)
legend('LM','FwdBck 3 Neuron');
xlabel('Number of Epochs');
ylabel('Sum Squared Error');
title('Convergence Comparison of LM to Back Propagating Neural Networks');

%% Testing Phase
for i = trainingIndex
    x = tp(i,:);
    sigmah = w1*[x 1]';
    uh = logsig(sigmah);
    sigmao = w2*uh;
    uo = sigmao;
    error = (d0(i) - uo);
end