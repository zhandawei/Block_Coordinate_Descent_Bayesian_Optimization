clearvars;clc;close all;
% objective function
fun_name = 'Ellipsoid';
% number of variables
num_vari = 100;
% lower and upper bounds
lower_bound = -5.12*ones(1,num_vari);
upper_bound = 5.12*ones(1,num_vari);
% batch size
batch_size = 5;
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
fprintf('Parallel BCD-BO on %d-D %s, batch size: %d, iteration: %d, evaluation: %d, best: %0.4g\n',num_vari,fun_name,batch_size,iteration-1,evaluation,fmin);
while evaluation < max_evaluation
    % train GP models
    GP_model = GP_train(sample_x,sample_y,lower_bound,upper_bound,1,0.01,100);
    % randomly divide all variables into q blocks
    order = randperm(num_vari);
    rand_index = [0,sort(randperm(num_vari-1,batch_size-1)),num_vari];
    block_size = rand_index(2:end) - rand_index(1:end-1);
    infill_x  = zeros(batch_size,num_vari);
    temp_x  = best_x;
    % get the q query points sequentially 
    for ii = 1:batch_size
        optimized_dim = order(sum(block_size(1:ii-1))+1:sum(block_size(1:ii)));
        [optimal_x,EI] = Optimizer_GA(@(x)-Infill_CoEI(x,GP_model,fmin,temp_x,optimized_dim),block_size(ii),lower_bound(optimized_dim),upper_bound(optimized_dim),10*block_size(ii),20);
        temp_x(optimized_dim) = optimal_x;
        infill_x(ii,:) = temp_x;
    end
    % evaluate q query points in parallel 
    infill_y = feval(fun_name,infill_x);
    iteration = iteration + 1;
    sample_x = [sample_x;infill_x];
    sample_y = [sample_y;infill_y];
    [fmin,ind]= min(sample_y);
    fmin_record(iteration,1) = fmin;
    best_x = sample_x(ind,:);
    evaluation = evaluation + size(infill_x,1);
    fprintf('Parallel BCD-BO on %d-D %s, batch size: %d, iteration: %d, evaluation: %d, best: %0.4g\n',num_vari,fun_name,batch_size,iteration-1,evaluation,fmin);
end
