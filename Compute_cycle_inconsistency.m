%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Copyright & License Notice
%% Cycle-Sync is copyrighted by Regents of the University of Minnesota and
%% the Regents of the University of California and covered by US 63/903,432. 
%% Regents of the University of Minnesota and the Regents of the University
%% of California will license the use of Cycle-Sync solely for 
%% educational and research purposes by non-profit institutions and US 
%% government agencies only. For other proposed uses, contact umotc@umn.edu.
%% The software may not be sold or redistributed without prior approval. 
%% One may make copies of the software for their use provided that the 
%% copies, are not sold or distributed, are used under the same terms and
%% conditions. As unestablished research software, this code is provided on
%% an "as is" basis without warranty of any kind, either expressed or implied.
%% The downloading, or executing any part of this software constitutes an 
%% implicit agreement to these terms. These terms and conditions are subject
%% to change at any time without prior notice.
%%------------------------------------------------
%% Compute cycle inconsistency for translation vectors
%%------------------------------------------------
%% Input Parameters: 
%% Ind: edge_num by 2 "edge indices matrix". Each row is the index of an edge (i,j) that is sorted as (1,2), (1,3), (1,4),... (2,3), (2,4),.... 
%% edge_num is the number of edges.
%% tijMat: 3 by edge_num matrix that stores the given relative translations corresponding to Ind
%% nsample: the number of sampled cycles per edge

%% Output:
%% out: Struct for relevant outputs
function[out] = Compute_cycle_inconsistency(Ind,tijMat,nsample)


            
    % building the graph   
    Ind_i = Ind(:,1);
    Ind_j = Ind(:,2);
    n=max(Ind,[],'all');
    m=size(Ind_i,1);
    AdjMat = sparse(Ind_i,Ind_j,1,n,n); % Adjacency matrix
    AdjMat = full(AdjMat + AdjMat');
        
    % initialization   
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
    
    ConvFlag = true;
    out.ConvFlag = ConvFlag;
    out.S0Mat = S0Mat;
    out.IndMat = IndMat;
    out.CoIndMat = CoIndMat;
    out.IndPos = IndPos;
return



