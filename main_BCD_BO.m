clearvars;clc;close all;
% objective function
fun_name = 'Ellipsoid';
% number of variables
num_vari = 100;
% lower and upper bounds
lower_bound = -5.12*ones(1,num_vari);
upper_bound = 5.12*ones(1,num_vari);
% number of initial design points
num_initial = 2*num_vari;
% number of maximum evaluations
max_evaluation = 10*num_vari;
% initial design
sample_x = lhsdesign(num_initial,num_vari,'criterion','maximin','iterations',1000).*(upper_bound-lower_bound)+lower_bound;
sample_y = feval(fun_name,sample_x);
evaluation =  size(sample_x,1);
iteration = 1;
[fmin,ind]= min(sample_y);
fmin_record(iteration,1) = fmin;
best_x = sample_x(ind,:);
% initialize block size
block_size = num_vari;
fprintf('BCD-BO on %d-D %s, block_size: %d,  iteration: %d, evaluation: %d, best: %0.4g\n',num_vari,fun_name,block_size,iteration-1,evaluation,fmin);
while evaluation < max_evaluation
    % train the GP model
    GP_model = GP_train(sample_x,sample_y,lower_bound,upper_bound,1,0.01,100);
    % randomly divide all variables with the block size
    num_group = ceil(num_vari/block_size);
    order = randperm(num_vari);
    EI = zeros(1,num_vari);
    infill_x = best_x;
    % optimize the acquisition function block by block
    for ii = 1:num_group
        start_idx = (ii-1)*block_size + 1;
        end_idx = min(ii*block_size, num_vari);  
        optimized_dim = order(start_idx:end_idx);
        [optimal_x, EI] = Optimizer_GA(@(x)-Infill_EI(x,GP_model,fmin,infill_x,optimized_dim),...
            length(optimized_dim), lower_bound(optimized_dim), upper_bound(optimized_dim),10*length(optimized_dim), 20);
        infill_x(optimized_dim) = optimal_x;
    end
    % evaluate the new solution
    infill_y = feval(fun_name,infill_x);
    %  adjust the block size
    if infill_y >= fmin
        block_size = max(block_size-1,1);
    end
    iteration = iteration + 1;
    sample_x = [sample_x;infill_x];
    sample_y = [sample_y;infill_y];
    [fmin,ind]= min(sample_y);
    fmin_record(iteration,1) = fmin;
    best_x = sample_x(ind,:);
    evaluation = evaluation + size(infill_x,1);
    fprintf('BCD-BO on %d-D %s, block_size: %d,  iteration: %d, evaluation: %d, best: %0.4g\n',num_vari,fun_name,block_size,iteration-1,evaluation,fmin);
end








