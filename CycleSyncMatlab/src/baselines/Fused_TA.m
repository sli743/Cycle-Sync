% Copyright (C) 2024 Computer Vision Lab, Electrical Engineering, 
% Indian Institute of Science, Bengaluru, India.
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
% 
%     * Redistributions of source code must retain the above copyright
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above
%       copyright notice, this list of conditions and the following
%       disclaimer in the documentation and/or other materials provided
%       with the distribution.
%     * Neither the name of Indian Institute of Science nor the
%       names of its contributors may be used to endorse or promote products
%       derived from this software without specific prior written permission.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
% OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
% THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
% FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
% DEALINGS IN THE SOFTWARE.
% 
% Author: Lalit Manam
%
% This file is part of the implementation for:
% Lalit Manam and Venu Madhav Govindu, Fusing Directions and Displacements in Translation Averaging, 
% International Conference on 3D Vision, 2024 

% This script is modified from the source:
% https://bbzh.github.io/document/BATA.zip


function [t,ed_ret_idx]=Fused_TA(tij_index,tij_observe,param,t_init_given)
% Inputs::
% tij_index: 2 by n matrix specifying the edge (i,j)
% tij_observe: 3 by n matrix specifying tij for each edge (i,j), such that
% tj-ti/norm(tj-ti) = tij
% param: Parameters for the method
% t_init_given: 3 by m matrix of camera translation initialization
% idxConst: Camera index which is chosen as origin
% Outputs::
% t: 3 by m matrix specifying the camera translations
% ed_ret_idx: indexes of the edge retained

    if(nargin<4)
        t_init_given=[];
    end

    G=graph(tij_index(1,:),tij_index(2,:));
    D=degree(G);
    [~,idxConst]=max(D); 
        
    numofcam = max(max(tij_index));
    numofobser = size(tij_observe,2);
    Ri_T = repmat(eye(3),numofobser,1);
    % Rj_T = repmat(eye(3),numofobser,1);
    Rj_T=Ri_T;
    
    index_ti_I = [(1:3*numofobser)' (1:3*numofobser)' (1:3*numofobser)'];   % position of coefficient for Ri
    index_ti_J = (tij_index(1,:)-1)*3+1;
    index_ti_J = [index_ti_J index_ti_J+1 index_ti_J+2];
    index_ti_J = index_ti_J(ceil((1:3*size(index_ti_J,1))/3), :);
    
    %index_tj_I = [(1:3*numofobser)' (1:3*numofobser)' (1:3*numofobser)'];   % position of coefficient for Rj
    index_tj_I = index_ti_I;
    index_tj_J = (tij_index(2,:)-1)*3+1;                 
    index_tj_J = [index_tj_J index_tj_J+1 index_tj_J+2];
    index_tj_J = index_tj_J(ceil((1:3*size(index_tj_J,1))/3), :);
    
    At0_full = sparse(index_ti_I,index_ti_J, Ri_T,3*numofobser,3*numofcam)...
       -sparse(index_tj_I,index_tj_J,Rj_T,3*numofobser,3*numofcam);
    
    Z=zeros(4);
    
    Aeq1 = sparse(reshape(repmat(1:numofobser,3,1),[],1),1:3*numofobser,tij_observe)*At0_full;
    Aeq1 = sum(Aeq1); 
    beq1 = numofobser; 
    Aeq2 = repmat(eye(3),1,numofcam); 
    beq2 = zeros(3,1); 
    Aeq = [Aeq1;Aeq2];
    beq = [beq1;beq2];
    
    % Setting up initialization
    if(isempty(t_init_given))
        Svec = rand(1,numofobser)+0.5;
        Svec = Svec/sum(Svec)*numofobser;
        S = reshape(repmat(Svec,3,1),[],1);
        W = ones(3*numofobser,1);
    else
        t=t_init_given;
        t=t-mean(t,2);
        t=t(:);
        curr_scale_fac=Aeq1*t;
        t=t*beq1/curr_scale_fac; % Dot product constraint
        
        Aij = reshape(At0_full*t,3,numofobser);
        tij_T = reshape(tij_observe,3,numofobser);
        Svec = sum(Aij.*tij_T)./sum(tij_T.*tij_T);
        tmp3 = repmat(Svec,[3,1]);
        S = tmp3(:);
        tmp = reshape(At0_full*t-S.*tij_observe(:),3,[]);
        Wvec = (sum(tmp.*tmp) + param.delta).^(-0.5); % (L1 loss)
        W = reshape(repmat(Wvec,[3,1]),[],1); % Computing initial weights
    end
    
    %% %% RLUD-Init    
    ii = 1; errPrev=1; errCurr=0;
    while(ii<=param.numofiterinit && abs(errPrev-errCurr)/errPrev>param.relerrthreinit)
        A = sparse(1:3*numofobser,1:3*numofobser,W)*At0_full;
        B = W.*S.*tij_observe(:);
        X = [(A'*A) Aeq'; Aeq Z]\[(A'*B); beq];
        t = X(1:3*numofcam);
        Aij = reshape(At0_full*t,3,numofobser);
        tij_T = reshape(tij_observe,3,numofobser);
        Svec = sum(Aij.*tij_T)./sum(tij_T.*tij_T);
        Svec(Svec<0)=0;
        tmp3 = repmat(Svec,[3,1]);
        S = tmp3(:);
        errPrev=errCurr;
        tmp= sum(reshape(At0_full*t-S.*tij_observe(:),3,[]).^2);
        Wvec = (tmp + param.delta).^(-0.5); %(L1 loss)
        errCurr=sum(sqrt(tmp));
        W = reshape(repmat(sqrt(Wvec),[3,1]),[],1);
        ii = ii + 1;
    end    
    
    %% %% Fused TA
    ii=1; % Number of outer iterations
    node_ret_idx=true(3,numofcam);    
    errPrevRLUD=1; errCurrRLUD=0; errPrevBATA=1; errCurrBATA=0; % Errors of different cost functions
    tprev=inf(size(t));
    t_rlud=zeros(size(t)); %t_bata=t_rlud;
    tij_T = reshape(tij_observe,3,numofobser);
    
    Aeq1_full = sparse(reshape(repmat(1:numofobser,3,1),[],1),1:3*numofobser,tij_observe)*At0_full;
    Aeq1_full = sum(Aeq1_full); 
    
    while(ii<=param.numofouteriter && ...
            norm(tprev(node_ret_idx(:))-t(node_ret_idx(:)))>param.relchangethreouter && ...
            (abs(errPrevRLUD-errCurrRLUD)/errPrevRLUD)>param.relerrthreouter && ...
            (abs(errPrevBATA-errCurrBATA)/errPrevBATA)>param.relerrthreouter)
        tprev=t;
        errPrevRLUD=errCurrRLUD;
        errPrevBATA=errCurrBATA;
        
        %% RLUD step
        Aij=reshape(At0_full*t,3,numofobser);
        Svec_rlud=sum(Aij.*tij_T)./sum(tij_T.*tij_T); % RLUD scales
        S=repmat(Svec_rlud,[3,1]); S(S<0)=0;  
        ResErrRLUD=sqrt(sum(reshape((At0_full*t-S(:).*tij_T(:)).^2,3,[])));
        errCurrRLUD=sum(ResErrRLUD,'omitnan');                
        
        [ed_ret_idx,node_ret_idx]=extractLargestConnComp(Svec_rlud,tij_index,node_ret_idx);
        numEdgesRet=sum(ed_ret_idx(1,:)); 
        numNodesRet=sum(node_ret_idx(1,:)); 
        
        Svec_red=Svec_rlud(ed_ret_idx(1,:));
        At0_full_red=At0_full(ed_ret_idx(:),node_ret_idx(:));
        tij_red=tij_T(:,ed_ret_idx(1,:));
        tmpSc=repmat(Svec_red,[3,1]);
        
        Wvec=1./(1+(ResErrRLUD.*ResErrRLUD)/param.robustthreRLUD^2); % Cauchy loss
        Wvec_red=Wvec(ed_ret_idx(1,:));
        W=reshape(repmat(Wvec_red,[3,1]),[],1);        
        
        Ar=sparse(1:3*numEdgesRet,1:3*numEdgesRet,sqrt(W))*At0_full_red;
        Br=sqrt(W).*tmpSc(:).*tij_red(:);
                
        Aeq1=sparse(reshape(repmat(1:numEdgesRet,3,1),[],1),1:3*numEdgesRet,tij_red)*At0_full_red;
        Aeq1=sum(Aeq1); 
        beq1=numEdgesRet; 
        Aeq2=repmat(eye(3),1,numNodesRet); 
        beq2=zeros(3,1); 
        Aeq=[Aeq1;Aeq2];
        beq=[beq1;beq2];
                
        X = [2*(Ar'*Ar) Aeq'; Aeq zeros(size(Aeq,1))]\[2*(Ar'*Br); beq];
        t_rlud(:)=nan;
        t_rlud(node_ret_idx(:))=X(1:3*numNodesRet);
        
        % Uncertainty based fusion in online fashion
        % Fusing RLUD Solution with previous solution as RLUD prior
        t_prev=tprev;

        t_rlud(~node_ret_idx(:))=nan; 
        t_prev(~node_ret_idx(:))=nan; 
        
        t_rlud_mean=mean(reshape(t_rlud,3,[]),2,'omitnan');
        t_rlud=t_rlud-repmat(t_rlud_mean,numofcam,1);
        
        t_prev_mean=mean(reshape(t_prev,3,[]),2,'omitnan');
        t_prev=t_prev-repmat(t_prev_mean,numofcam,1);
        
        nodes=find(node_ret_idx(1,:));
        ed_ret_idx=repmat(ismember(tij_index(1,:),nodes)&ismember(tij_index(2,:),nodes),3,1);
        numEdgesRet=sum(ed_ret_idx(1,:)); % No. of retained edges
        
        curSc=sum(Aeq1_full'.*t_rlud,'omitnan');
        t_rlud=t_rlud*numEdgesRet/curSc;
        
        curSc=sum(Aeq1_full'.*t_prev,'omitnan');
        t_prev=t_prev*numEdgesRet/curSc;

        node_ret_idx_temp=node_ret_idx; node_ret_idx_temp(:,idxConst)=false;
        t_rlud=t_rlud-repmat(t_rlud(3*idxConst-2:3*idxConst),numofcam,1);
        t_prev=t_prev-repmat(t_prev(3*idxConst-2:3*idxConst),numofcam,1);
                
        t_rlud_temp=t_rlud(node_ret_idx_temp(:));
        t_prev_temp=t_prev(node_ret_idx_temp(:));

        % Compute the Hessian
        At0_full_red=At0_full(ed_ret_idx(:),node_ret_idx_temp(:));
        
        % % RLUD
        Aij=reshape(At0_full*t_rlud,3,numofobser);
        Svec_rlud=sum(Aij.*tij_T)./sum(tij_T.*tij_T); 
        S=repmat(Svec_rlud,[3,1]); S(S<0)=0;
        ResErrRLUD=sqrt(sum(reshape((At0_full*t_rlud-S(:).*tij_T(:)).^2,3,[])));
        Wvec=1./(1+(ResErrRLUD.*ResErrRLUD)/param.robustthreRLUD^2); % Cauchy loss
        Wvec_red=Wvec(ed_ret_idx(1,:));
        W=reshape(repmat(Wvec_red,[3,1]),[],1);
        Ar=sparse(1:3*numEdgesRet,1:3*numEdgesRet,sqrt(W))*At0_full_red;
        H_rlud=Ar'*Ar; 

        % % RLUD prior        
        Aij=reshape(At0_full*t_prev,3,numofobser);
        Svec_rlud=sum(Aij.*tij_T)./sum(tij_T.*tij_T); 
        S=repmat(Svec_rlud,[3,1]); S(S<0)=0;
        ResErrRLUD=sqrt(sum(reshape((At0_full*t_prev-S(:).*tij_T(:)).^2,3,[])));
        Wvec=1./(1+(ResErrRLUD.*ResErrRLUD)/param.robustthreRLUD^2); % Cauchy loss
        Wvec_red=Wvec(ed_ret_idx(1,:));
        W=reshape(repmat(Wvec_red,[3,1]),[],1);
        Ar=sparse(1:3*numEdgesRet,1:3*numEdgesRet,sqrt(W))*At0_full_red;
        H_rlud_prev=Ar'*Ar;

        % Fuse the solutions
        t_fused_temp = (H_rlud+H_rlud_prev)\(H_rlud*t_rlud_temp+H_rlud_prev*t_prev_temp);

        % Update the translations
        t(:)=nan;
        t(node_ret_idx_temp(:))=t_fused_temp;
        t(3*idxConst-2:3*idxConst)=0;
        
        %% BATA Step
        Aij=reshape(At0_full*t,3,numofobser);
        Svec_bata=sum(Aij.*tij_T)./sum(Aij.*Aij); % BATA scales
        S=repmat(Svec_bata,[3,1]); S(S<0)=0; 
        A=sparse(1:numofobser*3,1:numofobser*3,S(:),numofobser*3,numofobser*3)*At0_full;
        ResErrBATA=sqrt(sum(reshape((A*t-tij_T(:)).^2,3,[])));
        errCurrBATA=sum(ResErrBATA,'omitnan');        
        % Cauchy loss
        Wvec=1./(1+((ResErrBATA/param.robustthreBATA).^2));
        
        % Inner loop for BATA for recomputing scales
        t_bata=t;
        jj =1; tprevIn=inf(size(t_bata));
        while(jj<=param.numofinneriter && norm(tprevIn(node_ret_idx(:))-t_bata(node_ret_idx(:)))>param.relchangethreinner)
            
            tprevIn=t_bata;
                        
            [ed_ret_idx,node_ret_idx]=extractLargestConnComp(Svec_bata,tij_index,node_ret_idx);
            node_ret_idx_temp=node_ret_idx; node_ret_idx_temp(:,idxConst)=false;
            Svec_red=Svec_bata(ed_ret_idx(1,:));
            At0_full_red=At0_full(ed_ret_idx(:),node_ret_idx_temp(:));
            tij_red=tij_observe(:,ed_ret_idx(1,:));
            tmpSc=repmat(Svec_red,[3,1]);
            Wvec_red=Wvec(ed_ret_idx(1,:));
            W=reshape(repmat(Wvec_red,[3,1]),[],1);
            
            % Cost function
            Ar=sparse(1:length(W),1:length(W),sqrt(W).*tmpSc(:),length(W),length(W))*At0_full_red;
            Br=sqrt(W).*tij_red(:);
            
            % Solve for translations
            X=(Ar'*Ar)\(Ar'*Br);
            t_bata(:)=nan;
            t_bata(node_ret_idx_temp(:))=X;
            t_bata(3*idxConst-2:3*idxConst)=[0;0;0];
            
            % Compute scales
            Aij=reshape(At0_full*t_bata,3,numofobser);
            Svec_bata=sum(Aij.*tij_T)./sum(Aij.*Aij);
            
            % Update inner loop variables
            jj=jj+1;
            
        end        

        % Uncertainty based fusion in online fashion
        % Fusing BATA Solution with previous solution as BATA prior
        
        t_=t;
        
        t_(~node_ret_idx(:))=nan; 
        t_bata(~node_ret_idx(:))=nan; 
        
        t__mean=mean(reshape(t_,3,[]),2,'omitnan');
        t_=t_-repmat(t__mean,numofcam,1);
        
        t_bata_mean=mean(reshape(t_bata,3,[]),2,'omitnan');
        t_bata=t_bata-repmat(t_bata_mean,numofcam,1);
        
        nodes=find(node_ret_idx(1,:));
        ed_ret_idx=repmat(ismember(tij_index(1,:),nodes)&ismember(tij_index(2,:),nodes),3,1);
        numEdgesRet=sum(ed_ret_idx(1,:)); 
        
        curSc=sum(Aeq1_full'.*t_,'omitnan');
        t_=t_*numEdgesRet/curSc;
        
        curSc=sum(Aeq1_full'.*t_bata,'omitnan');
        t_bata=t_bata*numEdgesRet/curSc;

        node_ret_idx_temp=node_ret_idx; node_ret_idx_temp(:,idxConst)=false;
        t_=t_-repmat(t_(3*idxConst-2:3*idxConst),numofcam,1);
        t_bata=t_bata-repmat(t_bata(3*idxConst-2:3*idxConst),numofcam,1);
                
        t__temp=t_(node_ret_idx_temp(:));
        t_bata_temp=t_bata(node_ret_idx_temp(:));
        
        % Compute the Hessian        
        At0_full_red=At0_full(ed_ret_idx(:),node_ret_idx_temp(:));
        
        % % BATA prior       
        Aij=reshape(At0_full*t_,3,numofobser);
        Svec_bata=sum(Aij.*tij_T)./sum(Aij.*Aij); 
        S=repmat(Svec_bata,[3,1]); S(S<0)=0; 
        A=sparse(1:numofobser*3,1:numofobser*3,S(:),numofobser*3,numofobser*3)*At0_full;
        ResErrBATA=sqrt(sum(reshape((A*t_-tij_T(:)).^2,3,[])));
        Wvec=1./(1+((ResErrBATA/param.robustthreBATA).^2)); % Cauchy loss
        Wvec_red=Wvec(ed_ret_idx(1,:));
        W=reshape(repmat(Wvec_red,[3,1]),[],1);
        Svec_red=Svec_bata(ed_ret_idx(1,:));
        tmpSc=repmat(Svec_red,[3,1]);
        Ar=sparse(1:length(W),1:length(W),sqrt(W).*tmpSc(:),length(W),length(W))*At0_full_red;
        H_bata_prior=Ar'*Ar; 
        
        % % BATA
        Aij=reshape(At0_full*t_bata,3,numofobser);
        Svec_bata=sum(Aij.*tij_T)./sum(Aij.*Aij); 
        S=repmat(Svec_bata,[3,1]); S(S<0)=0; 
        A=sparse(1:numofobser*3,1:numofobser*3,S(:),numofobser*3,numofobser*3)*At0_full;
        ResErrBATA=sqrt(sum(reshape((A*t_bata-tij_T(:)).^2,3,[])));
        Wvec=1./(1+((ResErrBATA/param.robustthreBATA).^2)); % Cauchy loss
        Wvec_red=Wvec(ed_ret_idx(1,:));
        W=reshape(repmat(Wvec_red,[3,1]),[],1);
        Svec_red=Svec_bata(ed_ret_idx(1,:));
        tmpSc=repmat(Svec_red,[3,1]);
        Ar=sparse(1:length(W),1:length(W),sqrt(W).*tmpSc(:),length(W),length(W))*At0_full_red;
        H_bata=Ar'*Ar;
        
        % Fuse the solutions
        t_fused_temp = (H_bata_prior+H_bata)\(H_bata_prior*t__temp+H_bata*t_bata_temp);
        
        % Update the translations
        t(:)=nan;
        t(node_ret_idx_temp(:))=t_fused_temp;
        t(3*idxConst-2:3*idxConst)=0;

        %% Update outer loop variables
        ii=ii+1;
        
    end
    
    %% Update variables to be sent
    t=reshape(t,3,[]);
    nodes=find(node_ret_idx(1,:));
    ed_ret_idx=ismember(tij_index(1,:),nodes)&ismember(tij_index(2,:),nodes);
    
end

function [ed_ret_idx,node_ret_idx]=extractLargestConnComp(Svec,tij_index,node_ret_idx)
    ed_ret_idx=true(3,size(tij_index,2));
    ed_ret_idx(:,Svec<=0)=false; 
    nodes=find(node_ret_idx(1,:));
    eidx=ismember(tij_index(1,:),nodes)&ismember(tij_index(2,:),nodes);
    ed_ret_idx(:,~eidx)=false; 
    edges=tij_index(:,ed_ret_idx(1,:));    
    
    G=graph(edges(1,:),edges(2,:));
    bins=conncomp(G,'OutputForm','vector');
    nodes = find(bins==mode(bins));
    ed_ret_idx_temp=ismember(tij_index(1,:),nodes)&ismember(tij_index(2,:),nodes)&ed_ret_idx(1,:);
    ed_ret_idx=repmat(ed_ret_idx_temp,[3,1]);
    node_ret_idx(:)=false;
    node_ret_idx(:,nodes)=true;
end
