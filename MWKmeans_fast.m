function [U, W, Z, UDistToZ, LoopCount, D] = MWKmeans_fast(Data, K, p, InitialCentroids, InitialW)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Minkowski Weighted K-Means (fast = Minkowski centre approximation)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%From:
%Amorim, R.C. and Mirkin, B., Minkowski Metric, Feature Weighting and Anomalous Cluster Initialisation in K-Means Clustering, 
%Pattern Recognition, vol. 45(3), pp. 1061-1075, 2012.
%
%Parameters:
%Data
%      Dataset, format: Entities x Features
%K
%      Total number of clusters in the dataset
%p
%      Minkowski and weight exponent 
%InitialCentroids (optional)
%      Initial centroids the algorithm should use, format: K x Features. 
%InitialW (Optional)
%      Initial set of weights the algorithm should use, format: K x
%      Features.
%
%Outputs
%
%U
%      Cluster Labels. Clearly they may not directly match the dataset
%      labels (you should use a confusion matrix).
%W
%      Final Weights
%Z
%      Final Centroids
%UDistToZ
%      The distance of each entity to its centroid.
%LoopCount
%      The number of loops the algorithm took to converge. The maximum is
%      hard-coded to 500 (variable MaxLoops)

%M Lines and N Columns
[M,N] = size(Data);
MaxLoops = 500; %just in case, shouldn't be necessary

%Binary Variable M x k, cluster label
U = zeros(M, 1);
OldU = U;

% Step 1
%Get Random Initial Centroids (VALUES) X Number of K
if exist('InitialCentroids','var')
    Z = InitialCentroids;
else
    %rng(sum(100*clock));
    %Z=datasample(Data,K, 'replace', false);
    Z=Data(randperm(M,K),:);
end

%Generate Initial set of weights
if exist('InitialW','var') 
    W = InitialW;
else
    W(1:K,1:N)=1/N;
end

LoopCount = 0;
while LoopCount<=MaxLoops
    %Step 2
    %Find Initial U that is minimized for the initials Z and W
    [NewUtp1, UDistToZ]= GetNewU (Data, Z, W.^p, p,M, K);
    %If there is no alteration in the labels stop
    if isequal(NewUtp1, U), break, end;
    
    %if the labes are equal to the previous-previous lables - stop (cycle)
    %This shouldn't happen
    if isequal(NewUtp1, OldU), break; end;
    
    %Step 3
    OldU = U;
    U = NewUtp1;
    %Get New Centroids
    Ztp1 = GetNewZ(Data, U, K,p, Z);
    %If there is no alteration in the centroids stop
    if isequal(Ztp1,Z), break, end;
    Z = Ztp1;
    %Step 4
    %Update the Weights
    [W,D] = GetNewW(Data, U, Z, p,N, K);
    LoopCount = LoopCount + 1;
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GetNewU calculates the labels (clustering)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [U, UDistToZ]= GetNewU (Data, Z, W, p, M, K)
temp_distances=zeros(M,K);
%for each Centroid, gets the distance, line = entity, col = cluster
for k = 1 : K
    temp_distances(:,k) = sum(bsxfun(@times,abs(bsxfun(@minus,Data,Z(k,:))).^p, W(k,:)),2);    
end
[UDistToZ, U]=min(temp_distances,[],2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GetNewZ calculates the centroids
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Z = GetNewZ(Data, U, K,p, OldZ)
Z = OldZ;
for l = 1 : K
    if sum(U==l)>1
        %if there isnt any entity in the cluster (shouldnt be the case!) = dont change the Z
        Z(l,:)=New_cmt(Data(U==l,:),p);
    end
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%GetNewW calculates the new set of weights
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [W,D] = GetNewW(Data, U, Z, p,N, K)
D = zeros(K,N);
W=D;
%Calculates the dispersion of each feature at each cluster
for k = 1 : K
    D(k,:) = sum(abs(bsxfun(@minus,Data(U==k,:),Z(k,:))).^p,1);
end
D = D + mean(mean(D));
%Calculates the actual feature weights
if p~=1
    exp = 1/(p-1);
    for l = 1 : K
        for j = 1 : N
            tmpD=D(l,j);
            W(l,j)= 1/sum((tmpD(1,ones(N,1))./D(l,:)).^exp);
        end
    end
else
    for l = 1 : K
        [~, MinIndex] = min(D(l,:));
        W(l,MinIndex)=1;
    end
end 






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% New_cmt calculates the Minkowski centre of the data at the Minkowski
% Approximation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [DataCenter]=New_cmt(Data,p)
N=size(Data,1);
if N==1
    DataCenter = Data;
    return;
elseif p<=1.5
    DataCenter = median(Data);
    return;
else
    DataCenter = mean(Data);
    return;
end
