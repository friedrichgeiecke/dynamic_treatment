% Code for: DYNAMICALLY OPTIMAL TREATMENT ALLOCATION USING
% REINFORCEMENT LEARNING
% by Karun Adusumilli, Friedrich Geiecke, and Claudio Schilter

% This file: obtain the static "Kitagawa-Tetenov"-Policy as benchmark without
% age

% This code is heavily based on the replication file for 
% "Who Should Be Treated?
%  Welfare Maximization Methods for Treatment Choice"
% by Toru Kitagawa and Aleksey Tetenov
% We even operate it within the replication folder offered by Kitagawa and Tetenov

% The key changes are to impose a budget constraint and to use our rewards instead of theirs.

% Interestingly, this optimizer seems to work much better with unique
% X-data (Kitagawa and Tetenov do use Xu, the unique X entries). We
% therefore also use Xu. This implies that the budget constraint (that
% one-quarter of the total people can be treated) is a little more
% involved. We simply used manual trial and error to find a constraint on 
% Xu that translates into the desired constraint on the full X.

% Prerequisite: transformed_rct_data_d_rob_ols_Xfit.csv from python run

clear all
%need to add paths to integrate cplex studio optimizer into matlab
addpath('C:\Program Files\IBM\ILOG\CPLEX_Studio1210\cplex\matlab')
addpath('C:\Users\ClaudioSchilter\Documents\IBM\ILOG\CPLEX_Studio1210\cplex\examples\src\matlab')
addpath('C:\Program Files\IBM\ILOG\CPLEX_Studio1210\cplex\matlab\x64_win64')

cd 'C:\Users\ClaudioSchilter\Dropbox\Reinforcement_learning\Kitagawa_Tetenow_Replication_files'

rng(0); % Starting point for the RNG
bsimax = 0; % number of bootstrap reps, 0 = no bootstrap
cost = true; %take treatment costs into account
if (cost)
    diary('jtpa_linear_cost.log');
    tcost = 774;
    filename_coefs = 'jtpa_linear_cost_coefs.mat';
    filename_results = 'jtpa_linear_cost_results.mat';
else
    diary('jtpa_linear.log');
    tcost = 0;
    filename_coefs = 'jtpa_linear_coefs.mat';
    filename_results = 'jtpa_linear_results.mat';
end

% Setup CPLEX optimization options
opt = cplexoptimset('cplex');
opt.parallel = 1;
opt.threads = 4;
opt.simplex.tolerances.feasibility = 1e-08;
opt.mip.tolerances.integrality = 1e-08;
opt.mip.strategy.variableselect = 3;
opt.mip.strategy.nodeselect = 2;
opt.mip.strategy.lbheur = 1;
opt.mip.limits.cutpasses = -1;
opt.display = 'off';

% DATA INPUT AND PREPARATION
load('jtpa_kt.mat')

Y = earnings-tcost.*D;
meanY = mean(Y);
Y = Y - meanY; % demean outcomes
Yscale = max(abs(Y));
Y = Y./Yscale; % rescale demeaned outcomes to [-1,1]
n = length(Y); % sample size
ps = 2/3; % propensity score
% create regressor matrix X from the data
covariates  = [edu prevearn];

% Rescale covariates to [-1,1]
Xscale = ones(n,1)*max(abs(covariates));
X = [ones(n,1) covariates./Xscale];

%our rewards
g_helper = table2array(readtable('transformed_rct_data_d_rob_ols_Xfit.csv'));
g = g_helper(:,5);

k = size(X,2);

vhats_p = zeros(bsimax,1); 
vhats_n = zeros(bsimax,1); 
% first run - main estimation using the original sample
bsperm = [1:n]';
gn = g(bsperm,:);
Xn = X(bsperm,:);
tic
for bsi = 0:bsimax
    % DRAW BOOTSTRAP SAMPLE WITH INDEXES bsperm
    gr = g(bsperm,:);
    Xr = X(bsperm,:);
    % Subtract Wn from Wbs
    if (bsi>0)
        gr = [gr; -gn];
        Xr = [Xr; Xn];
    end
    % Compress data with identical X's
    [Xr, Ind] = sortrows(Xr);
    gr = gr(Ind);
    Xu = unique(Xr,'rows');
    %Xu = Xr
    nu  = size(Xu,1); % number of unique covariate vectors
    gu = zeros(nu,1);
    jj = 1;
    for j = 1:size(Xr,1)
        if ~(sum(Xr(j,:)~=Xu(jj,:)))
            gu(jj) = gu(jj) + gr(j);
        else
            jj = jj+1;
            gu(jj) = gu(jj) + gr(j);
        end
    end

  
    sameedu = (Xu(1:end-1,2)==Xu(2:end,2));
    % decreasing in pre-program earnings
    Mineq_d = [diag(sameedu) zeros(nu-1,1)] + [zeros(nu-1,1) diag(-sameedu)];

    % increasing in pre-program earnings
    Mineq_i = [diag(-sameedu) zeros(nu-1,1)] + [zeros(nu-1,1) diag(sameedu)];
  
    
    f = [zeros(k,1); -gu]; % objective function coefficients
    f_n = [zeros(k,1); gu]; % Reverse objective function coefficients
    B = 1; % bounds on coefficients
    C = B*sum(abs(Xu),2); % maximum values of x'beta

    minmargin = max(1,C)*(1e-8); 
    Aineq_d = [[-Xu diag(C)]; [Xu -diag(C)]; [zeros(nu-1,k) Mineq_d]; [zeros(1,k) ones(1,nu)]];
   
    
    %budget constraint (obtained by trial and error. Note: there are
    %considerably fewer unique observations if age is not a covariate as well)
    xx=389;
    
    Aineq_i = [[-Xu diag(C)]; [Xu -diag(C)]; [zeros(nu-1,k) Mineq_i]; [zeros(1,k) ones(1,nu)]];
    bineq = [[C-minmargin];[-minmargin]; minmargin(1:nu-1,:); xx];

    
    lb = [-B*ones(k,1); zeros(nu,1)];
    ub = [ B*ones(k,1);  ones(nu,1)];
    
    % Variable type string
    ctype = strcat(repmat('C',1,k),repmat('B',1,nu));

    % Welfare maximization with treatment decreasing in earnings
    [sol_pd, v_pd] = cplexmilp(f,Aineq_d,bineq,[],[],[],[],[],lb,ub,ctype,[],opt);

    [sol_pi, v_pi] = cplexmilp(f,Aineq_i,bineq,[],[],[],[],[],lb,ub,ctype,[],opt);
    
    if (bsi==0)
        if (v_pd < v_pi)
            beta = sol_pd(1:k,:);
            v    = -v_pd;
        else
            beta = sol_pi(1:k,:);
            v    = -v_pi;
        end
    else
        vhats_p(bsi) = -min(v_pd,v_pi);
        [sol_nd, v_nd] = cplexmilp(f_n,Aineq_d,bineq,[],[],[],[],[],lb,ub,ctype,[],opt);
        [sol_ni, v_ni] = cplexmilp(f_n,Aineq_i,bineq,[],[],[],[],[],lb,ub,ctype,[],opt);
        vhats_n(bsi) = -min(v_nd,v_ni);
    end
    
    bsperm = randi(n,[n 1]);
    
    % progress markers
    bsi
    toc
end
writematrix(beta,'beta_kt_final_noage_389.csv');


diary off
