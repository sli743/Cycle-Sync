%% Author: Shaohan Li
%% © Regents of the University of Minnesota. All rights reserved
%%------------------------------------------------
%% Cycle-Edge Message Passing for Translation Synchronization
%%------------------------------------------------
%% Input Parameters: 
%% Ind: edge_num by 2 "edge indices matrix". Each row is the index of an edge (i,j) that is sorted as (1,2), (1,3), (1,4),... (2,3), (2,4),.... 
%% edge_num is the number of edges.
%% tijMat: 3 by edge_num matrix that stores the given relative directions corresponding to Ind
%% CEMP_parameters.max_iter: the number of iterations of CEMP
%% CEMP_parameters.reweighting: the sequence of reweighting parameter beta_t
%% CEMP_parameters.nsample: the number of sampled cycles per edge

%% Output:
%% SVec: Estimated corruption levels of all edges



function[SVec,out] = CEMP_translation(Ind,tijMat,CEMP_parameters)
    %CEMP parameters
    T=CEMP_parameters.max_iter; 
    beta_cemp=CEMP_parameters.reweighting;
    nsample = CEMP_parameters.nsample;
    T_beta = length(beta_cemp);
    if T_beta<T
        % if the reweighting parameter vector is short, then the rest of
        % missing elements are set to constant
        beta_cemp = [beta_cemp,beta_cemp(end)*(ones(1,T-T_beta))]; 
    end
    if ~isfield(CEMP_parameters,'no_cemp_iters')
        CEMP_parameters.no_cemp_iters = false;
    end
            
    % building the graph   
    Ind_i = Ind(:,1);
    Ind_j = Ind(:,2);
    n=max(Ind,[],'all');
    m=size(Ind_i,1);
    AdjMat = sparse(Ind_i,Ind_j,1,n,n); % Adjacency matrix
    AdjMat = full(AdjMat + AdjMat');
        
    % start CEMP iterations as initialization   
    disp('sampling 3-cycles')
    % Matrix of codegree:
    % CoDeg(i,j) = 0 if i and j are not connected, otherwise,
    % CoDeg(i,j) = # of vertices that are connected to both i and j
    CoDeg = (AdjMat*AdjMat).*AdjMat;
    AdjPos = AdjMat;
    % label positive codegree elements as -1
    AdjPos(CoDeg>0)=-1;
    AdjPosLow = tril(AdjPos);
    AdjPosLow(AdjPosLow==0)=[];
    % find 1d indices of edges with positive codegree
    IndPos = find(AdjPosLow<0);
    IndPosbin = zeros(1,m);
    IndPosbin(IndPos)=1;
    % CoIndMat(:,l)= triangles sampled that contains l-th edge
    % e.g. l=(3,5), then CoIndMat(:,l)=[2,9,8,...] means that...
    % triangles 352, 359, 358,... are sampled
    for l = IndPos
        i = Ind_i(l); j = Ind_j(l);
       CoIndMat(:,l)= datasample(find(AdjMat(:,i).*AdjMat(:,j)), nsample);
    end

    disp('Sampling Finished!')
    disp('Initializing')

    tijMat4d = zeros(3,n,n);
    for l = 1:m
        i=Ind_i(l);j=Ind_j(l);
        tijMat4d(:,i,j)=tijMat(:,l); % store relative translations in 3xnxn tensor
        tijMat4d(:,j,i)=-tijMat(:,l);
        IndMat(i,j)=l; % construct edge index matrix (for 2d-to-1d index conversion)
        IndMat(j,i)=-l;
    end
   
    % start computing cycle-inconsistency
    tki0 = zeros(3,m,nsample); % Rki0(:,:,l,s) is Rki if the s-th sampled cycle of edge ij (whose 1-d index is l) is ijk
    tjk0 = zeros(3,m,nsample);
    tij0 = zeros(3,m,nsample);
    for l = IndPos
        tki0(:,l,:) = tijMat4d(:,CoIndMat(:,l), Ind_i(l));
        tjk0(:,l,:) = tijMat4d(:,Ind_j(l),CoIndMat(:,l));
        tij0(:,l,:) = repmat(tijMat4d(:,Ind_i(l), Ind_j(l)),[1,1,nsample]);
    end
 
    % reshape above matrices for easier multiplication
    tki0Mat = reshape(tki0,[3,m*nsample]);
    tjk0Mat = reshape(tjk0,[3,m*nsample]);
    tij0Mat = reshape(tij0, [3,m*nsample]);
    
    t_cycle = tki0Mat + tjk0Mat + tij0Mat;
    t_len = (reshape(sqrt(t_cycle(1,:).^2+t_cycle(2,:).^2+t_cycle(3,:).^2), [m,nsample]))'; % R_trace(l,s) stores Tr(Rij*Rjk*Rki) for that cycle
    S0Mat = t_len;   % S0Mat(l,s) stores d(Rij*Rjk*Rki, I) for that cycle. d is the normalized geodesic distance.
    SVec = mean(S0Mat,1); % initialized corruption level estimates s_{ij,0}
    SVec(~IndPosbin)=1; % If there is no 3-cycle for that edge, then set its sij as 1 (the largest possible).
    
    if CEMP_parameters.no_cemp_iters
        ConvFlag = true;
        out.ConvFlag = ConvFlag;
        out.S0Mat = S0Mat;
        out.IndMat = IndMat;
        out.CoIndMat = CoIndMat;
        out.IndPos = IndPos;
        return
    end


    disp('Initialization completed!')  
    disp('Reweighting Procedure Started ...')

   for iter = 1:T     
        % parameter controling the decay rate of reweighting function
        beta = beta_cemp(iter);
        Ski = zeros(nsample, m);
        Sjk = zeros(nsample, m);
        for l = IndPos
            i = Ind_i(l); j=Ind_j(l);
            Ski(:,l) = SVec(abs(IndMat(i,CoIndMat(:,l))));
            Sjk(:,l) = SVec(abs(IndMat(j,CoIndMat(:,l))));
        end
        Smax = Ski+Sjk;
        % compute cycle-weight matrix (nsample by m)
        WeightMat = exp(-beta*Smax);
        weightsum = sum(WeightMat,1);
        % normalize so that each column sum up to 1
        WeightMat = bsxfun(@rdivide,WeightMat,weightsum);
        SMat = WeightMat.*S0Mat;
        % sij at current iteration
        SVec0 = SVec;
        SVec = sum(SMat,1);
        SVec(~IndPosbin)=1;
        SVec(isnan(SVec)) = 1;
        diffS = norm(SVec0 - SVec);
        fprintf('SVec diff is %d\n',diffS);
        fprintf('Reweighting Iteration %d Completed!\n',iter)   
   end
        disp('Completed!')
        
    if diffS<1
        ConvFlag=true;
    else
        ConvFlag=false;
    end
    out.ConvFlag = ConvFlag;
end

