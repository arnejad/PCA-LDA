% This code was developed by Ali Rahmani Nejad for educational purposes
clear all;

%% Intializations
A = xlsread('breast_preprocessed.xls');
A(47294,:) = [];
LABLES = zeros(1, 128);
LABLES = A(47294, :);
A(47294,:) = [];

%% Part 1 - PCA

% subtracting mean from all data
meanAll = mean(A,2);
data = (A - meanAll)';

% run SVD, only U and S are needed
[U,S,~] = svd(data, 'econ');
reducedData = U * S;

covMat = cov(reducedData);  % producing covariance matrix

[V,D] = eig(covMat);   % eigen value of 
D=diag(D);              % all eigen values

% Sorting eigenvectors based on corresponding eigenvalues
loadingMatrix = zeros(size(V, 2), size(V, 2));   % matrix of the number of eigen vectors
for i = 1:size(V, 2)
    loadingMatrix(:, i) = V(:,find(D==max(D)));    % eigenvector of the max eigen value
    V(:, find(D==max(D)))=[];
    D(find(D==max(D))) = [];
end

scoreMatrix = loadingMatrix * reducedData;

% draw score plot
subplot(3,1,1);
% plot(scoreMatrix(:,1), scoreMatrix(:,2), 'o');
scatter(scoreMatrix(:, 1), scoreMatrix(:, 2), 110, LABLES, 'filled');
title('Score Plot')
xlabel('PC1')
ylabel('PC2')

% draw Loading plot for
subplot(3,1,2);
% loadingMatrix = loadingMatrix .* (10^20);
plotv((loadingMatrix(:, 1:2)))
hold on
scatter(loadingMatrix(:, 1),loadingMatrix(:, 2), 'filled');
hold on
% dx = 2; dy = 2; % displacement so the text does not overlay the data points
a = [1:128]'; b = num2str(a); c = cellstr(b);
text(loadingMatrix(:,1), loadingMatrix(:,2), c);
title('Loading Plot')
xlabel('PC1')
ylabel('PC2')

%draw Cumulative variance
subplot(3,1,3);
rowCumVar = cumsum(var(scoreMatrix'.').');
bar(rowCumVar);
title('Cumulative variance ')
xlabel('PCs') 
ylabel('varriance') 

clear  a A b c covMat D dx dy eigenVectors i k rowCumVar U V


%% Part 2 - LDA

% dividing data into two seporate classes

luminalClass = [];
nonLuminalClass = [];

for i=1:128
    if LABLES(i)==1
        luminalClass = [luminalClass; reducedData(i,:)];
    else
        nonLuminalClass = [nonLuminalClass; reducedData(i,:)];
    end
end

% extracting 14 samples from each class for testing purposes
testLuminal = luminalClass(size(luminalClass,1)-13:size(luminalClass,1), :);
luminalClass(size(luminalClass,1)-13: size(luminalClass,1), :) = [];

testNonLuminal = nonLuminalClass(size(nonLuminalClass,1)-13:size(nonLuminalClass,1), :);
nonLuminalClass(size(nonLuminalClass,1)-13: size(nonLuminalClass,1), :) = [];


% Our implementation of LDA
luminalClassMean = mean(luminalClass, 1);
nonLuminalClassMean = mean(nonLuminalClass,1);

% calculating Sw
Si1 = zeros(128,128);
 for i=1:size(luminalClass,1)  % sum of rows products
      Si1 = Si1+ (luminalClass(i,:)-luminalClassMean)' * (luminalClass(i,:)-luminalClassMean);
 end

Si2 = zeros(128,128);
 for i=1:size(nonLuminalClass,1)
      Si2 = Si2+ (nonLuminalClass(i,:)-nonLuminalClassMean)' * (nonLuminalClass(i,:)-nonLuminalClassMean);
 end
 
Sw = Si1 + Si2;

totalMean = mean([luminalClass;nonLuminalClass],1);

Sb = zeros(128,128);
Sb = Sb + size(luminalClass,1) * (luminalClassMean - totalMean)' * (luminalClassMean - totalMean);
Sb = Sb + size(nonLuminalClass,1) * (nonLuminalClassMean - totalMean)' * (nonLuminalClassMean - totalMean);

SwInv = pinv(Sw);
A = SwInv * Sb;
[V,D] = eig(A);       %extracting eigenvectors/values
D=diag(D);              % all eigen values

% sorting eigenVectors
W = zeros(size(V, 2), size(V, 2));   % matrix of the number of eigen vectors
for i = 1:size(V, 2)
    W(:, i) = V(:,find(D==max(D)));    % eigenvector of the max eigen value
    V(:, find(D==max(D)))=[];
    D(find(D==max(D))) = [];
end

finalLuminal = luminalClass * W;
finalNonLuminal = nonLuminalClass * W;

        % TESTING PHASE
classLumMean = mean(finalLuminal,1);
classNonLumMean = mean(finalNonLuminal,1);

rightGuesses = 0;
for i=1:size(testLuminal,1)
    
   if sqrt(sum((classLumMean - testLuminal(i, :)).^2, 2)) <  sqrt(sum((classNonLumMean - testLuminal(i, :)).^2, 2))
        rightGuesses = rightGuesses + 1;
   end
end

for i=1:size(testNonLuminal,1)
    
   if sqrt(sum((classNonLumMean - testNonLuminal(i, :)).^2, 2)) <  sqrt(sum((classLumMean - testNonLuminal(i, :)).^2, 2))
        rightGuesses = rightGuesses + 1;
   end
end
fprintf('Number of right predictions for our model: %f\n', rightGuesses);
