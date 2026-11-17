function FE = mwk_FE(Data, K, r, type)
%Feature extraction using mwk-means

%Type = 1
%              Re-scales PCA
%Type = 2 
%              Re-scales NMF
%Type = 3
%               Re-scales ICA


%First run all mwk-means
U=cell(1,0);
W=cell(1,0);
P = [];
all_p = [1.1, 1.5, 2, 3];
for p_idx = 1 : size(all_p,2)
    p = all_p(1, p_idx);
    best_obj = inf;
    for i = 1 : 50
        [tmp_U, tmp_W, ~, UDistToZ] = MWKmeans_fast(Data, K, p);
        obj = sum(UDistToZ);
        if obj < best_obj
            best_obj = obj;
            best_tmp_U = tmp_U;
            best_tmp_W = tmp_W;
            best_tmp_p = p;
        end
    end
    U{1,size(U,2)+1} = best_tmp_U;
    W{1,size(W,2)+1} = best_tmp_W;
    P = [P;best_tmp_p];
end
%Find the clustering with the highest adjusted rand index to all others
n_experiments = size(U,2);
similarity = zeros(n_experiments, n_experiments);
for i = 1: n_experiments -1
    for j = i+1:n_experiments 
        similarity(i,j) = RandIndex(U{1,i}, U{1,j});
        similarity(j, i) = similarity(i,j);
    end
end
[~,idx]=max(sum(similarity));
Best_U = U{1,idx};
Best_W = W{1,idx};
Best_p = P(idx,1);

%Rescale
for k = 1 : max(Best_U)
    Data(Best_U == k, :) = Data(Best_U == k, :).*(Best_W(k,:));
end

if type == 1
    %PCA
    [~, score] = pca(Data);
    FE = score(:,1:r);
elseif type==2
    %NMF
    non_neg_Data = (Data + abs(min(Data)));
    [FE, ~] = nnmf(non_neg_Data, r, 'replicates', 50);
elseif type==3
    %ICA
    best_err = inf;
    for i = 1:50
        warning('off', 'all');
        Mdl_tmp = rica(Data, r, 'IterationLimit', 100);
        warning('on', 'all');
        FE_tmp = transform(Mdl_tmp, Data);
        err = norm(Data - FE_tmp * Mdl_tmp.TransformWeights', 'fro');
        if err < best_err
            best_err = err;
            FE = FE_tmp;
        end
    end
end

