clear all
clc
close all

load 'TrainingSamplesDCT_8_new.mat'

%% PART A

% Initially we have to setup the priors
py_G = 1053/(1053+250);
py_F = 250/(1053+250);

% loading the dataset for background and foreground
C = 8; 
BG =  TrainsampleDCT_BG;
FG =  TrainsampleDCT_FG;



% Solving for h and updating the parameters for 100 times for the Foreground.
mix_BG_5 = {};
for mix = 1:5
    
    % initialize 8 parameters for background 
    pbg = 1-0.5*rand(1,8);
    pbg = pbg/sum(pbg);
    mbg = {};
    sbg = {};

    for i = 1:C
        mbg{i} = 1 - 0.5*rand(1,64);
        sbg{i} = diag(1 - 0.5*rand(1,64));
    end 
    
    % Loop through iteration to optimize the parameters.
    for i = 1:100

        conditional_BG = 0;
        for j = 1:C
            conditional_BG =  conditional_BG + (mvnpdf(BG,mbg{j},sbg{j}))*pbg(j); 
        end

        Hij = [];
        for j = 1:C
            Hij = [Hij (mvnpdf(BG,mbg{j},sbg{j}))*pbg(j)./conditional_BG]; 
        end 

        pbg = sum(Hij)/(i+1);
        m = (Hij'*BG)./sum(Hij)';
        s =(Hij'*((BG - mbg{j}).^2))./sum(Hij)';
        for j = 1:C
            mbg{j} = m(j,:); 
            sbg{j} = diag(s(j,:));
        end

    end
    
    mix_BG_5{mix} = {pbg, mbg, sbg}; 
    
end

save('bg_data.mat', 'mix_BG_5')

% Solving for h and updating the parameters for 100 times for the Foreground.
mix_FG_5 = {};
for mix =  1:5
    
    % initialize 8 parameters for foreground 
    pfg = 1 - 0.5*rand(1,8);
    pfg = pfg/sum(pfg);
    mfg = {};
    sfg = {};

    for i = 1:C
        mfg{i} = 1 - 0.5*rand(1,64);
        sfg{i} = diag(1 - 0.5*rand(1,64));
    end 
    
    % Loop through iteration to optimize the parameters.
    for i = 1:100

        conditional_FG = 0;
        for j = 1:C
            conditional_FG =  conditional_FG + (mvnpdf(FG,mfg{j},sfg{j}))*pfg(j); 
        end

        Hij = [];
        for j = 1:C
            Hij = [Hij (mvnpdf(FG,mfg{j},sfg{j}))*pfg(j)./conditional_FG]; 
        end 

        pfg = sum(Hij)/(i+1);
        m = (Hij'*FG)./sum(Hij)';
        s =(Hij'*((FG - mfg{j}).^2))./sum(Hij)';
        for j = 1:C
            mfg{j} = m(j,:); 
            sfg{j} = diag(s(j,:));
        end

    end
    mix_FG_5{mix} = {pfg, mfg, sfg};
    
end
save('fg_data.mat', 'mix_FG_5')

