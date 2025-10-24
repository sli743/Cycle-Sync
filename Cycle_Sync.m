%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% Input: AdjMat (img_num by img_num adjacency matrix), 
%% tijMat (3 by edge_num matrix that stores the pairwise directions),
%% opts (struct that stores the relevant parameters)
%% Output: t_est (estimated absolute camera locations)
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [t_est, out] = Cycle_Sync(AdjMat, tijMat, opts)

t_start = tic;
%% General algorithmic parameters:
%%
if ~isfield(opts, 'tolQuad');       opts.tolQuad = 1e-8;   end    % Inner loop tolerance
if ~isfield(opts, 'delt');          opts.delt = 1e-12;     end    % IRLS regularization parameter
if ~isfield(opts, 'maxit');         opts.maxit = 200;      end    % Maximum number of iterations
if ~isfield(opts, 'maxitQuad');     opts.maxitQuad = 200;  end    % Maximum number of inner iterations
if ~isfield(opts, 'staglim');       opts.staglim = 5;      end    % Number of iteration for stagnation
if ~isfield(opts, 'sinmin');       opts.sinmin = 0.6;      end    % Parameter for sine truncation
if ~isfield(opts, 'tau4');       opts.tau4 = 4;      end    % parameter for reweighting function
if ~isfield(opts, 'flam');       opts.flam = @(x) x/(x+10);      end    % reweighting function for location refinement
if ~isfield(opts, 'tau1');       opts.tau1 = 20;      end    % reweighting function for location refinement
if ~isfield(opts, 'WLSiters');       opts.WLSiters = 20;      end    % reweighting function for location refinement
if ~isfield(opts, 'beta');       opts.beta = 20;      end    % parameter beta

tolQuad      = opts.tolQuad;
delt         = opts.delt;
maxit        = opts.maxit;
maxitQuad    = opts.maxitQuad;
staglim      = opts.staglim;
WLSiters     = opts.WLSiters;
sinmin = opts.sinmin; % Truncation of angle sine for AAB statistic
tau4 = opts.tau4;
flam = opts.flam;

n = size(AdjMat,1); % number of cameras
m = size(tijMat,2); % number of edges
nsample = 200; % number of sample 3-cycles for each edge
tau1 = opts.tau1; % Reweighting Parameter for weight init
beta = opts.beta; % Reweighting Parameter for exponential/Welsch reweighting
niter = 5; % number of iterations for AAB
% 2d indices of edges, i<j
[Ind_j, Ind_i] = find(tril(AdjMat,-1)); 
tijMat3d = zeros(3,n,n);
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

% store pairwise directions in 3 by n by n tensor
% construct edge index matrix (for 2d-to-1d index conversion)
IndMat = zeros(n,n);
for l = 1:m
    i=Ind_i(l);j=Ind_j(l);
    tijMat3d(:,j,i)=tijMat(:,l);
    tijMat3d(:,i,j)=-tijMat(:,l);
    IndMat(i,j)=l;
    IndMat(j,i)=l;
end

    disp('Sampling Triangles...')

% CoIndMat(:,l)= triangles sampled that contains l-th edge
% e.g. l=(3,5), then CoIndMat(:,l)=[2,9,8,...] means that...
% triangles 352, 359, 358,... are sampled

for l = IndPos
    i = Ind_i(l); j = Ind_j(l);
    CoInds = find(AdjMat(:,i).*AdjMat(:,j));
    Xkis = tijMat3d(:,Ind_i(l),CoInds);
    Xjks = tijMat3d(:,Ind_j(l),CoInds);
    cosines = Xkis.*Xjks;
    cosines = abs(sum(cosines,1));
    CoInds_good = CoInds(cosines<sqrt(1-sinmin^2));
    if ~isempty(CoInds_good)
        error('No good edges for T-AAB. Consider reducing opts.sinmin. ')
    end
    CoIndMat(:,l)= datasample(CoInds, nsample);
end

    disp('Triangle Sampling Finished!')
    disp('Computing Naive AAB ...')

Xki = zeros(3,m,nsample);
Xjk = zeros(3,m,nsample);
for l = IndPos
    Xki(:,l,:) = tijMat3d(:,Ind_i(l),CoIndMat(:,l));
    Xjk(:,l,:) = -tijMat3d(:,Ind_j(l),CoIndMat(:,l));
end
% Xki stores gamma_ki of all triangles ijk
% Xki has nsample blocks. Each block is 3 by m (m gamma_ki's)
% i corresponds to edge (i,j), k corresponds to a sampled triangle
Xki = reshape(Xki,[3,m*nsample]);
% Xjk stores gamma_jk of all triangles ijk
% Xjk has nsample blocks. Each block is 3 by m (m gamma_jk's)
% j corresponds to edge (i,j), k corresponds to a sampled triangle
Xjk = reshape(Xjk,[3,m*nsample]);
% Compute Naive AAB statistic using the AAB formula
% If l-th edge is (i,j), then X(k,l) is the dot product between
% gamma_ij and gamma_ki. Y and Z are similar
X = (reshape(sum(Xki.*kron(ones(1,nsample),tijMat),1),[m,nsample]))';
Y = (reshape(sum(Xjk.*kron(ones(1,nsample),tijMat),1),[m,nsample]))';
Z = (reshape(sum(Xki.*Xjk,1),[m,nsample]))';
S = 1.0*(X<(Y.*Z)).*(Y<(X.*Z));
% AAB formula in matrix form
SAABMat0 = abs(acos(S.*(X.^2+Y.^2-2*X.*Y.*Z)./(1-Z.^2)+(S-1.0).*min(X,Y)));
% Taking average for each column to obtain the Naive AAB for each edge
IRAABVec = mean(SAABMat0,1);

    disp('Naive AAB Computed!')

% compute maximal/minimal AAB inconsistency
maxAAB = max(max(SAABMat0));
minAAB = min(min(SAABMat0));

    disp('Reweighting Procedure Started ...')

tau = 1; tau_rate = 2;tau_max = 20;
for iter = 1:niter
    % parameter controling the decay rate of reweighting function
    tau = min(tau*tau_rate,tau_max);
    Ski = zeros(nsample, m);
    Sjk = zeros(nsample, m);
    for l = IndPos
        i = Ind_i(l); j=Ind_j(l);
        Ski(:,l) = IRAABVec(IndMat(i,CoIndMat(:,l)));
        Sjk(:,l) = IRAABVec(IndMat(j,CoIndMat(:,l)));
    end
    Smax = Ski+Sjk;
    % compute weight matrix (nsample by m)
    WeightMat = exp(-tau*Smax);
    weightsum = sum(WeightMat,1);
    % normalize so that each column sum up to 1
    WeightMat = bsxfun(@rdivide,WeightMat,weightsum);
    SAABMat = WeightMat.*SAABMat0;
    % IR-AAB at current iteration
    IRAABVec = sum(SAABMat,1);

        fprintf('Reweighting Iteration %d Completed!\n',iter)   

end

    disp('Completed!')
out.IRAABVec = IRAABVec;

n = size(AdjMat,1);
d = size(tijMat,1);




[Ind_j, Ind_i] = find(tril(AdjMat,-1));
ss_num = length(Ind_i);
j_Vec_Lmat = [Ind_i Ind_j]';
j_Vec_Lmat = j_Vec_Lmat(:);
i_Vec_Lmat = kron([1:ss_num]',ones(2,1));
val_Vec_Lmat = kron(ones(ss_num,1),[1;-1]);
l_mat = sparse(i_Vec_Lmat,j_Vec_Lmat,val_Vec_Lmat,ss_num,n,2*ss_num);
Lmat = kron(l_mat,speye(d));
V = kron(ones(n,1),speye(d));

%% Matrix of cost function
Mmat = [Lmat -sparse([1:d*ss_num]',kron([1:ss_num]',ones(d,1)),tijMat(:),d*ss_num,ss_num,d*ss_num)];

%% Start IRLS loops
count = 1;
optsQuad = optimset('Algorithm','interior-point-convex','MaxIter',maxitQuad,...
    'TolFun',tolQuad,'Display','off');
stagcnt = 0;

wMat = kron(sparse([1:ss_num]',[1:ss_num]', exp(-tau1* IRAABVec), ss_num,ss_num,ss_num),speye(d));

% CEMP params
CEMP_parameters.max_iter = 30;
CEMP_parameters.reweighting = 20;
CEMP_parameters.nsample = 100;
CEMP_parameters.no_cemp_iters = true;

for iter1 = 1:WLSiters
    lam = flam(iter1);
    t_alph_est = quadprog(Mmat'*wMat*Mmat, sparse(d*n+ss_num,1), ...
        [-sparse(ss_num,d*n) -speye(ss_num)], -ones(ss_num,1),...
        [V' sparse(d,ss_num)], sparse(d,1),[],[],[],optsQuad);
    t_est = reshape(t_alph_est(1:d*n),d,n);

    if count == 1
        out.t_init = t_est;
    end
    alph = t_alph_est(d*n+1:end);
    alph=alph';
    % Update IRLS weights
    residual_vec = reshape(Mmat*t_alph_est,d,ss_num);
    residual_norms = sqrt(bsxfun(@dot,residual_vec,residual_vec));
    alpha_gamma=tijMat;
    alpha_gamma(1,:)=alph.*tijMat(1,:);
    alpha_gamma(2,:)=alph.*tijMat(2,:);
    alpha_gamma(3,:)=alph.*tijMat(3,:);
    tijhat = residual_vec + alpha_gamma;
    lenij = sqrt(sum(tijhat.*tijhat));
    tijMat0 = tijMat.*lenij;
    [~,out] = CEMP_translation([Ind_i,Ind_j],tijMat0,CEMP_parameters);
    nsample = CEMP_parameters.nsample;
    IndMat = out.IndMat;
    CoIndMat = out.CoIndMat;
    IndPos = out.IndPos;
    S0Mat = out.S0Mat;
    Ski = zeros(nsample, m);
    Sjk = zeros(nsample, m);
    for l = IndPos
        i = Ind_i(l); j=Ind_j(l);
        Ski(:,l) = residual_norms(abs(IndMat(i,CoIndMat(:,l))));
        Sjk(:,l) = residual_norms(abs(IndMat(j,CoIndMat(:,l))));
    end
    Smax = Ski+Sjk;
    % compute weight matrix (nsample by m)
    WeightMat = exp(-beta*Smax);
    weightsum = sum(WeightMat,1);
    WeightMat(:,weightsum<1e-16) = 1e-16;
    WeightMat(:,isnan(weightsum)) = 1e-16;
    weightsum = sum(WeightMat,1);
    weightsum = max(weightsum,1e-16);
    % normalize so that each column sum up to 1
    WeightMat = bsxfun(@rdivide,WeightMat,weightsum);
    SAABMat = WeightMat.*S0Mat;
    % IR-AAB at current iteration
    IRAABVec = sum(SAABMat,1);
    SVec2 = IRAABVec;

    ES_max = lam*SVec2 + (1-lam)*residual_norms;
    ES_max(ES_max > 10) = 10; %avoid numerical issue caused by 0 weight edges
    wv = exp(-tau4*ES_max)./(ES_max + delt);
    wMat = kron(sparse([1:ss_num]',[1:ss_num]', wv, ss_num,ss_num,ss_num),speye(d));
end


t_end = toc(t_start);

alph = t_alph_est(d*n+1:end);
out.alph = alph;
out.TotalTime = t_end;
if (iter > maxit)
    if (stagcnt <= staglim)
        out.flag = -2;
    else
        out.flag = -1;
    end
else
    out.flag = 0;
end

return



