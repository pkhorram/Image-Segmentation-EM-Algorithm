clc
clear all
close all

load 'bg_data.mat'
load 'fg_data.mat'
C = 8;

% Priors
py_G = 1053/(1053+250);
py_F = 250/(1053+250);

% Dimensions
d = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64];

%% Loading the IMAGE and implementation

% Loading the data
image = imread('cheetah.bmp');
image = im2double(image);
image2 = padarray(image,[4 4],0,'both');
load 'Zig-Zag Pattern.txt' 

% Implementation on the image.

rows = size(image,1);
columns = size(image,2);
pad = 4;
batch_size = 8;





ee =0;
for m = 2:5
    for n = 1:5
        ee = ee + 1;
        P = [];
        pbg = mix_BG_5{m}{1};
        mbg = mix_BG_5{m}{2};
        sbg = mix_BG_5{m}{3};

        pfg = mix_FG_5{n}{1};
        mfg = mix_FG_5{n}{2};
        sfg = mix_FG_5{n}{3};
        
        for T = 1:length(d)
           
            A = zeros(rows,columns);

            for r = 5 : 5+rows -1
                col = 5;
                while col <= columns 
                    block = image2([(r-pad):((r-pad) + batch_size - 1)],[(col-pad):((col-pad) + batch_size -1)]);
                    vec = dct2(block);
                    new_vec(Zig_Zag_Pattern(:)+1) = vec(:); 
                    new_vec = new_vec(1:d(T));

                    conditional_BG = 0;
                    for j = 1:C
                        conditional_BG =  conditional_BG + (mvnpdf(new_vec,mbg{j}(1:d(T)),sbg{j}(1:d(T),1:d(T))))*pbg(j); 
                    end
                    Gprob = py_G*conditional_BG; 


                    conditional_FG = 0;
                    for j = 1:C
                        conditional_FG =  conditional_FG + (mvnpdf(new_vec,mfg{j}(1:d(T)),sfg{j}(1:d(T),1:d(T))))*pfg(j); 
                    end     
                    Fprob = py_F*conditional_FG;


                    if(Fprob>Gprob)
                       A(r-pad, col-pad) = 255;
                    end
                    col = col + 1;    
                end
            end
            compare = imread('cheetah_mask.bmp'); 
            total_param = size(compare,1)*size(compare,2);
            P_error = ((total_param - sum(sum( A == compare)))/total_param)* 100;
            P = [P P_error]; 
        end
        figure(m)
        plot(d,P)
        hold on
        title(sprintf('Plot of Probabilty of Error vs. Dimension for Classifier %d',ee))
        xlabel('Dimension')
        ylabel('Error')
        legend('FG1','FG2','FG3','FG4','FG5')
        grid on
        box on
    end
end


    









