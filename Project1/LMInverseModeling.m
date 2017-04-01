clear variables
close all
clc

filename = 'TrainedFwdNetWeights.xlsx';
wf1 = xlsread(filename,'w1');
wf2 = xlsread(filename,'w2');
wf1 = wf1(:,1:3);

xtarget = -1:0.2:1;
ytarget = 0:0.2:2;

numInvInputs = 2;
numInvOutputs = 3;
numInvNeurons = 24;
numFwdNeurons = 24;
numFwdOutputs = 2;
epoch = 0;
targetError = 1e-4;
maxEpochs = 10000;
invError = Inf*ones(numInvOutputs,1);
fwdError = invError;
invEpochError = Inf;
fwdEpochError = Inf;
% invJ = [];
% J = [];
mu = 0.1;
beta = 10;

wi1 = rand(numInvNeurons,numInvInputs+1);
wi2 = rand(numInvOutputs,numInvNeurons);

tempwi1 = wi1;
tempwi2 = wi2;
deltao = 1;

[X,Y] = meshgrid(xtarget,ytarget);
halfi = length(X);
invTrainError = zeros(2*length(X),1);
fwdTrainError = invTrainError;

partialxwrty = zeros(1,numFwdNeurons);
partialywrtx = zeros(1,numFwdNeurons);

%% Enter the trainig loop
while epoch < maxEpochs && invEpochError > targetError
    epoch = epoch + 1;
    invJ = [];
    
    % Initial Forward Propagation through the inverse and forward network
    for i = 1:numel(X)
        x = [X(i) Y(i) 1];
        x2 = x(1:2);
        uhi = HiddenLayer(x,wi1);
        uoi = OutputLayer(uhi,wi2);
%         xh = [uoi;1]';
        uhf = HiddenLayer(uoi',wf1);
        uof = OutputLayer(uhf,wf2);
        
        %Back Propagate through the forward network
        d0 = computeKinematics(uoi);
        dummy = d0-uof;
        fwdTrainError(i) = dummy(1);
        fwdTrainError(halfi+i) = dummy(2);
        
        deltahf = HiddenBackProp(deltao,wf2,uhf);
        
        %Back Propagate through inverse network
        deltaoi = InvHiddenBackProp(deltahf,wf1,uoi);
        
        %Formulation of Forward Jacobian
        tempxf = deltahf(:,1)';
        tempyf = deltahf(:,2)';
        for j = 1:numel(deltahf(:,1))
            tempxf = [tempxf (deltahf(j,1)*uoi)'];
            tempyf = [tempyf (deltahf(j,2)*uoi)'];
        end
        tempxf = [tempxf uhf' partialxwrty];
        tempyf = [tempyf partialywrtx uhf'];
        
        temp(i,:) = tempxf;
        temp(halfi+i,:) = tempyf;
        fwdJ = temp;
        
        %Formulation of Inverse Jacobian
        
    end
    repeat = 1;
    fwdEpochError = fwdTrainError'*fwdTrainError;
    fwdPreviousError = fwdTrainError;
    fwdJJ = fwdJ'*fwdJ;
    [row,col] = size(fwdJJ);
    id = eye(row,col);
    while repeat
        
        
        
        
    end
    
    
    
    
    
    
    
    
    
    
end