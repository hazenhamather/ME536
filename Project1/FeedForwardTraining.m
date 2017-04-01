%% Initialization
%Hazen Hamather 
%ME536
%Homework 2
%Project 1 Levenberg-Marquardt Neural Network
clear variables
close all
clc

% tp = [0 0;1 0;0 1;1 1];
% d0 = [0;1;1;0];
a = -pi;
b = pi;
c = 0;
numTrainPairs = 100;
tp1 = (b-a).*rand(numTrainPairs,1) + a;
tp2 = (b-c).*rand(numTrainPairs,1) + c;
tp3 = (b-c).*rand(numTrainPairs,1) + c;
tp = [tp1 tp2 tp3];
% tp = tp(1,:);
temp = zeros(2*length(tp1),144);
numInputs = 3;
numOutputs = 2;
numNeurons = 24;
% bias = rand(numNeurons,1);
learningRate = [0.25;0.30];
momentum = [0.7;0.8];
deltaw1last = 0;
deltaw2last = 0;
epoch = 0;
targetError = 1e-4;
maxEpochs = 10000;
error = Inf*ones(numOutputs,1);
epochError = Inf;
J = [];
mu = 0.1;
beta = 10;
partialxwrty = zeros(1,numNeurons);
partialywrtx = zeros(1,numNeurons);
half = length(tp1);
trainError = zeros(2*length(tp1),1);


%Randomizing starting weights
w1 = rand(numNeurons,numInputs + 1);
w2 = rand(numOutputs,numNeurons);
% w1 = [0.8928    0.3758    0.5820    0.8251;
%     0.8028    0.9382    0.6248    0.7427;
%     0.1434    0.5117    0.1162    0.0075;
%     0.4094    0.3768    0.2369    0.7688;
%     0.1752    0.5817    0.9928    0.1592;
%     0.5167    0.0771    0.8544    0.5840;
%     0.0256    0.4235    0.9102    0.9859;
%     0.7826    0.7532    0.6685    0.0098;
%     0.6379    0.5558    0.7816    0.4300;
%     0.5308    0.1462    0.9123    0.4494;
%     0.1331    0.1945    0.1978    0.0718;
%     0.2736    0.9455    0.7170    0.4891;
%     0.9665    0.7735    0.2134    0.0870;
%     0.5177    0.7789    0.7099    0.7293;
%     0.4982    0.5488    0.7314    0.3323;
%     0.2853    0.6910    0.4445    0.4567;
%     0.3208    0.4173    0.8658    0.4754;
%     0.9269    0.7083    0.7724    0.4770;
%     0.8158    0.6630    0.1943    0.3777;
%     0.8313    0.0539    0.4290    0.9993;
%     0.9274    0.7896    0.0763    0.6494;
%     0.7632    0.2907    0.1725    0.0600;
%     0.7119    0.1101    0.5843    0.1807;
%     0.0117    0.8846    0.3271    0.3825];
% 
% w2 = [0.8099,0.5440,0.124395452260010,0.4457,...
%     0.6464,0.1383,0.7723,0.6178,...
%     0.0322,0.9595,0.3920,0.9109,...
%     0.3222,0.1099,0.5718,0.0950,...
%     0.1379,0.1528,0.6712,0.4937,...
%     0.5341,0.8357,0.6548,0.7219;...
%     0.4446,0.4674,0.2500,0.1884,...
%     0.2329,0.0517,0.7070,0.5439,...
%     0.5714,0.8297,0.8114,0.8105,...
%     0.6574,0.8272,0.2731,0.2249,...
%     0.1878,0.2779,0.1068,0.0627,...
%     0.5494,0.4569,0.9750,0.6466];
% w1 = [.1 .2 .3 .4 .5 .6 .7 .8 .9 .2 .4 .6 .8 .1 .3 .5 .7 .9 .3 .6 .2 .5 .9 .7;...
%     .1 .2 .3 .4 .5 .6 .7 .8 .9 .2 .4 .6 .8 .1 .3 .5 .7 .9 .3 .6 .2 .5 .9 .7];
tempw1 = w1;
tempw2 = w2;
% alpha = learningRate(1);
alpha = 0.65;
deltao = 1;

%% Enter the training loop
while epoch < maxEpochs && epochError > targetError
    epoch = epoch + 1;
%     disp(epochError)
%     disp(epoch)
    J = [];
    for i = 1:numTrainPairs
       %Forward Propagate
       x = [tp(i,:) 1];
       x2 = tp(i,:);
       uh1 = HiddenLayer(x,w1);
       uo = OutputLayer(uh1,w2);
       
       %Back Propagate
       d0 = computeKinematics(x);
       dummy = d0-uo;
       trainError(i) = dummy(1);
       trainError(half+i) = dummy(2);
       
       %deltah1(:,1) is deltahx, deltah1(:,2) is deltahy
       deltah1 = HiddenBackProp(deltao,w2,uh1);
       
       %Formulation of the Jacobian
       tempx = deltah1(:,1)';
       tempy = deltah1(:,2)';
       for j = 1:numel(deltah1(:,1))
           tempx = [tempx deltah1(j,1)*x2];
           tempy = [tempy deltah1(j,2)*x2];
       end
       tempx = [tempx uh1' partialxwrty];
       tempy = [tempy partialywrtx uh1'];
       
       temp(i,:) = tempx;
       temp(half+i,:) = tempy;
       J = temp;
    end
    repeat = 1;
    epochError = trainError'*trainError;
    previousError = trainError;
    JJ = J'*J;
    [row,col] = size(JJ);
    id = eye(row,col);
    while repeat
        %Step 4
        change = ((JJ + mu*id)^(-1))*J'*previousError;
        reshapedChange = reshape(change(numNeurons+1:4*numNeurons,1),[3,24]);
        
        %Step 5
        tempw1(:,end) = w1(:,end) + change(1:numNeurons);
        for i = 1:length(reshapedChange(1,:))
            tempw1(i,1:3) = w1(i,1:3) + reshapedChange(:,i)';
%         tempw1(:,1) = w1(:,1) + change(numNeurons+1:2*numNeurons);
%         tempw1(:,2) = w1(:,2) + change(2*numNeurons+1,3*numNeurons);
%         tempw1(:,3) = w1(:,3) + change(3*numNeurons+1,4*numNeurons);
        end
        tempw2(1,:) = w2(1,:) + change(4*numNeurons+1:5*numNeurons)';
        tempw2(2,:) = w2(2,:) + change(5*numNeurons+1:6*numNeurons)';
        
        %repeat fwd propagation and compute new error
        for i = 1:length(tp1)
            x = [tp(i,:) 1];
            uhtemp = HiddenLayer(x,tempw1);
            uotemp = OutputLayer(uhtemp,tempw2);
            
            %Compute Kinematics
            d0 = computeKinematics(x);
            
            %Find error
            dummy = d0-uotemp;
            trainError(i) = dummy(1);
            trainError(half+i) = dummy(2);
        end
        currentError = trainError'*trainError;
        
        %Step 6
        if currentError < epochError
            mu = mu/beta;
            w1 = tempw1;
            w2 = tempw2;
            repeat = 0;
            epochError = currentError;
            total(epoch) = epochError;
        elseif mu > norm(J)
            w1 = tempw1;
            w2 = tempw2;
            repeat = 0;
            epochError = currentError;
            total(epoch) = epochError;
        else
            mu = mu*beta;
        end
        
    end
    
end

%% Training Phase
uo = zeros(numTrainPairs,2);
d0 = zeros(numTrainPairs,2);
for i = 1:numTrainPairs
    x = [tp(i,:) 1];
    uh1 = HiddenLayer(x,w1);
    uo(i,:) = OutputLayer(uh1,w2);
    d0(i,:) = computeKinematics(x);
end

%% Plotting the Network
plot(d0(:,1),d0(:,2),'ro',uo(:,1),uo(:,2),'bx')
xlabel('X');
ylabel('Y');
title('Real Kinematics vs Trained LM Neural Network Predictions');
legend('Real Kinematics','LM Neural Network');

%% Write Weights to File
filename = 'TrainedFwdNetWeights.xlsx';
xlswrite(filename,w1,'w1');
xlswrite(filename,w2,'w2');
xlswrite(filename,total,'totalError');