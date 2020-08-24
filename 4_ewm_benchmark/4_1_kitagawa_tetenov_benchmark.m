% Code for: DYNAMICALLY OPTIMAL TREATMENT ALLOCATION USING
% REINFORCEMENT LEARNING
% by Karun Adusumilli, Friedrich Geiecke, and Claudio Schilter

% This file: code to obtain the static "Kitagawa-Tetenov"-Policy as benchmark

% This code is heavily based on the replication file for 
% "Who Should Be Treated?
%  Welfare Maximization Methods for Treatment Choice"
% by Toru Kitagawa and Aleksey Tetenov
% We even operate it within the replication folder offered by Kitagawa and Tetenov

% The key changes are to add age as additional covariate, to impose a
% budget constraint, and to use our rewards instead of theirs.

% Interestingly, this optimizer seems to work much better with unique
% X-data (Kitagawa and Tetenov do use Xu, the unique X entries). We
% therefore also use Xu. This implies that the budget constraint (that
% one-quarter of the total people can be treated) is a little more
% involved. We simply used manual trial and error to find a constraint on 
% Xu that translates into the desired constraint on the full X. 

% Prerequisite: transformed_rct_data_d_rob_ols_Xfit.csv and
% transformed_rct_data_Rlr1.csv from python runs (as rewards)
% also: age.csv (simply the sole age column of data_lpe2.csv (sorted by recid))

clear all
%need to add paths to integrate cplex studio optimizer into matlab
addpath('C:\Program Files\IBM\ILOG\CPLEX_Studio1210\cplex\matlab')
addpath('C:\Users\ClaudioSchilter\Documents\IBM\ILOG\CPLEX_Studio1210\cplex\examples\src\matlab')
addpath('C:\Program Files\IBM\ILOG\CPLEX_Studio1210\cplex\matlab\x64_win64')

cd 'C:\Users\ClaudioSchilter\Dropbox\Reinforcement_learning\Kitagawa_Tetenow_Replication_files'

rng(0); % Starting point for the RNG
bsimax = 0; % number of bootstrap reps, 0 = no bootstrap. We do not use bootstrapping
% still, we leave the bootstrapping infrastructure here, for the code to be
% as comparable as possible to Kitagawa & Tetenov.
cost = true; %take treatment costs into account or not - we always do
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

%AGE
age = table2array(readtable('age.csv'));

% create regressor matrix X from the data
covariates  = [age edu prevearn];
%NOTE age added in front

% Rescale covariates to [-1,1]
Xscale = ones(n,1)*max(abs(covariates));
X = [ones(n,1) covariates./Xscale];

% use our rewards now (uncomment according to dr or std ols rewards)
%dr
g_helper = table2array(readtable('transformed_rct_data_d_rob_ols_Xfit.csv'));

%ols
%g_helper = table2array(readtable('transformed_rct_data_Rlr1.csv'));


g = g_helper(:,5);
% number of variables (including constant)
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
    
 
    sameage = (Xu(1:end-1,2)==Xu(2:end,2));
    
    sameageedu = (Xu(1:end-1,2)==Xu(2:end,2) & Xu(1:end-1,3)==Xu(2:end,3));

    % decreasing in edu and pre-program earnings
    Mineq_dd = [[diag(sameage) zeros(nu-1,1)] + [zeros(nu-1,1) diag(-sameage)]; [diag(sameageedu) zeros(nu-1,1)] + [zeros(nu-1,1) diag(-sameageedu)]];

    % decreasing in edu and incr in pre-program earnings
    Mineq_di = [[diag(sameage) zeros(nu-1,1)] + [zeros(nu-1,1) diag(-sameage)]; [diag(-sameageedu) zeros(nu-1,1)] + [zeros(nu-1,1) diag(sameageedu)]];

    % decreasing in edu and pre-program earnings
    Mineq_id = [[diag(-sameage) zeros(nu-1,1)] + [zeros(nu-1,1) diag(sameage)]; [diag(sameageedu) zeros(nu-1,1)] + [zeros(nu-1,1) diag(-sameageedu)]];
    
    % increasing in edu and pre-program earnings
    Mineq_ii = [[diag(-sameage) zeros(nu-1,1)] + [zeros(nu-1,1) diag(sameage)]; [diag(-sameageedu) zeros(nu-1,1)] + [zeros(nu-1,1) diag(sameageedu)]];
    
    
    f = [zeros(k,1); -gu]; % objective function coefficients
    f_n = [zeros(k,1); gu]; % Reverse objective function coefficients
    B = 1; % bounds on coefficients
    C = B*sum(abs(Xu),2); % maximum values of x'beta
    % this is just the sum of each row (dimension 2) multiplied with B=1
    minmargin = max(1,C)*(1e-8); 
    
    %last row is the budget constraint.
    Aineq_dd = [[-Xu diag(C)]; [Xu -diag(C)]; [zeros(2*(nu-1),k) Mineq_dd]; [zeros(1,k) ones(1,nu)]];
    Aineq_di = [[-Xu diag(C)]; [Xu -diag(C)]; [zeros(2*(nu-1),k) Mineq_di]; [zeros(1,k) ones(1,nu)]];
    Aineq_id = [[-Xu diag(C)]; [Xu -diag(C)]; [zeros(2*(nu-1),k) Mineq_id]; [zeros(1,k) ones(1,nu)]];
    Aineq_ii = [[-Xu diag(C)]; [Xu -diag(C)]; [zeros(2*(nu-1),k) Mineq_ii]; [zeros(1,k) ones(1,nu)]];

    
    % last elemet is the budget constraint on Xu (derived via trial and
    % error such that at most one-quarter of the X end up being treated.
    % (the result is not smooth, there are "jumps". However, we found that
    % treating just more than one-quarter generally produces lower
    % welfare in the simulations where the budget is limited to treat at
    % most one-quarter)
    
    %for DR
    xx=1313;
    
    %for std OLS
    %xx=1348; 
    
    %this bineq is the vector put vis-a-vis the Aineq matrix.
    bineq = [[C-minmargin];[-minmargin]; minmargin; minmargin(1:(2*(nu-1)-length(minmargin)),:); xx];
  
   
    
    lb = [-B*ones(k,1); zeros(nu,1)];
    ub = [ B*ones(k,1);  ones(nu,1)];
    
    % Variable type string
    ctype = strcat(repmat('C',1,k),repmat('B',1,nu));
    %a string that is CCCBBBBBBBBBBBBBB...

    % Welfare maximization with treatment decreasing in earnings
    [sol_pdd, v_pdd] = cplexmilp(f,Aineq_dd,bineq,[],[],[],[],[],lb,ub,ctype,[],opt);
          
    [sol_pdi, v_pdi] = cplexmilp(f,Aineq_di,bineq,[],[],[],[],[],lb,ub,ctype,[],opt);
    [sol_pid, v_pid] = cplexmilp(f,Aineq_id,bineq,[],[],[],[],[],lb,ub,ctype,[],opt);

    
    [sol_pii, v_pii] = cplexmilp(f,Aineq_ii,bineq,[],[],[],[],[],lb,ub,ctype,[],opt);
    if (bsi==0)
        if (min([v_pdd v_pdi v_pid v_pii]) == v_pdd)
            beta = sol_pdd(1:k,:);
            v    = -v_pdd;
        elseif (min([v_pdd v_pdi v_pid v_pii]) == v_pdi)
            beta = sol_pdi(1:k,:);
            v    = -v_pdi;
        elseif (min([v_pdd v_pdi v_pid v_pii]) == v_pid)
            beta = sol_pid(1:k,:);
            v    = -v_pid;
        else
            beta = sol_pii(1:k,:);
            v    = -v_pii;
        end
        %so for the beta. the rule. just take first 3(k) of the x above.
    else
   
    end

    bsperm = randi(n,[n 1]);
    
    % progress markers
    bsi
    toc
end
% Save coefficients and other variables

writematrix(beta,'beta_dr_final_max025treated.csv');
%writematrix(beta,'beta_ols_final_max025treated.csv');



diary off
