% scriptDetwin_myData
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% performs a deterministic detwinning on top of CrystFEL data and employs of "A-B" approach
% some changes in merging the AmB and ApB blocks compared to version1 of the code.
% A. Hosseini, UWM, Feb-2018, March-2019
% Modified to better manage memorx and run on PSI RA cluster.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&  STEP(1)  &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&  
% % Loading the dataset and initializations:

clear 
close all

clarify_disputed_snapshots = true;
detwin_disputed_snapshots = false;

% blocksize = 100; % 500 CMC - good on Ra cluster
blocksize = 500; %1000;  % for doing block-wise correlation (blocksize < nS)

% lattice sizes for bR in group P6(3):
a = 62.95; % in Angstrom
b = 62.95; % in Angstrom
c = 111.7; % in Angstrom

% load the crystal data which include some possible twins
fileData = 'data_bR_dark_no_laser_int_DRL_SCL_nS34569_nBrg54391.mat';
load(fileData,'T_drl_scl', ...
              'M_drl_scl', ...
              'miller_h', ...
              'miller_k', ...
              'miller_l');
%CMC - And t_uniform for light data

T = T_drl_scl;
M = M_drl_scl;

[nS, nBrg] = size(T);
% file name to save twinned data
fileTWN = sprintf('data_bR_dark_no_laser_int_DRL_SCL_nS%d_nBrg%d_twinned.mat',...
                  nS, nBrg);

%%_________________________________________________________________________________________
 % Making a twin of the whole dataset (adapted from RS):
%CMC - Indexing transformation (h, k, l) -> (k, h, l)
% for i = 1:nBrg
%    innd = [miller_h(i) miller_k(i) miller_l(i)];
%    NwInd = find(miller_k == innd(1) & ...
%                 miller_h == innd(2) & ...
%                 miller_l == innd(3));
%    if length(NwInd)                                                        %#ok<ISMT>
%        NewIndConv(i,1) = NwInd;                                            %#ok<SAGROW>
%    end                          
% end
% 
% %CMC - Do not transform (h, 0, l)
% Indx = find(miller_k == 0) ;
% NewIndConv(Indx) = Indx;
% 
% %CMC - Link unilinked (h, k, l) sets to the last one (intensity = 0)
% NewIndConv(nBrg+1) = nBrg+1;
% NewIndConv(NewIndConv == 0) = nBrg+1;
% 
% %_________________________________________________________________________________________
% % % Setting an upper limit on the resolution and removing common pixels between snapshots 
% % % and twins: h=k and h=0 | k=0 (adapted from RS):
% Qind = [miller_h/a, ...
%         miller_h./(sqrt(3)*a) + 2*miller_k./(sqrt(3)*b),  ...
%         miller_l./c];
% Np = nBrg;
% Radius_Qs = sqrt(Qind(:,1).^2+Qind(:,2).^2+Qind(:,3).^2); % distance of every Q point to origin
% %ShellRadius =[0.0667,0.6667];   % proposed by RS
% %%%%% resolution limit: 1/0.5-1/0.05
% %Ind_ResRange = find(Radius_Qs(1:Np) > ShellRadius(1) & Radius_Qs(1:Np) < ShellRadius(2));
% 
% %%%%% Common pixels of twin and original
% Inds = [miller_h miller_k miller_l];
% Common_Reflec = [];
% for j = 1:nBrg
%     if (Inds(j,1) ~= 0 && Inds(j,2) ~= 0 && Inds(j,1) ~= Inds(j,2));
% %   if (Inds(j,1) == 0 | Inds(j,2) == 0 | Inds(j,1) == Inds(j,2)  );
%         Common_Reflec = [Common_Reflec, j];                                %#ok<AGROW>
%     end
% end
% 
% %%%%%% Union of both:
% %NewPix = intersect(Ind_ResRange,Common_Reflec'); % proposed by RS
% NewPix = Common_Reflec';            % AH (i.e., no change in detwin if RS proposal is off)
% 
% 
% % % Selecting the desired pixels and save new data
% T(:,end+1)=0; %CMC - adding one column, see line 61
%               %CMC - reflections for which no twin is found in the dataset
%               %CMC - point to this zero intensity.
% M(:,end+1)=0;
% 
% T_tw = T(:,NewIndConv);
% M_tw = M(:,NewIndConv);
% 
% T = T(:,NewPix);
% T_tw = T_tw(:,NewPix);
% M = M(:,NewPix);
% M_tw = M_tw(:,NewPix);
% 
% miller_h = miller_h(NewPix);
% miller_k = miller_k(NewPix);
% miller_l = miller_l(NewPix);
% 
% % N_frames = 120000; %CMC - Reduce dark dataset
% % T    = T(1:N_frames,:);
% % T_tw = T_tw(1:N_frames,:);
% % M    = M(1:N_frames,:);
% % M_tw = M_tw(1:N_frames,:);
% 
% save(fileTWN,'T', 'T_tw', ...
%              'M', 'M_tw', ...
%              'miller_h', 'miller_l', 'miller_k',...
%              '-v7.3');
% %CMC - And t_uniform for light data.
% 
% %&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&  STEP(2)  &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&  
% % % Computing Pearson correlations including the masks, AminusB and AplusB 
% 
% %%%%%%%%%%%%%%%% block-wise method for the correlation R12
% clearvars -except blocksize fileTWN % to preserve memory
% 
% load(fileTWN); %data_bR_dark_no_laser_int_DRL_SCL_nS127028_nBrg54391_twinned.mat 
% 
% filePath_AmB = './AmB/';
% filePath_ApB = './ApB/';
% 
% X1 = full(T');
% X2 = full(T_tw');
% M1 = full(M');
% M2 = full(M_tw');
% 
% size(X1)
% size(X2)
% 
% X1(isnan(X1)) = 0;
% X2(isnan(X2)) = 0;
% nS = size(X1,2)                                                            %#ok<NOPTS>
% %blocksize = 1000; % blocksize<nS 
% 
% %CMC
% clear 'X1' 'X2' 'M1' 'M2';
% 
% tic
% for i = 1:ceil(nS/blocksize)
%     i                                                                      %#ok<NOPTS>
%     
%     X1 = full(T');
%     X2 = full(T_tw');
%     M1 = full(M');
%     M2 = full(M_tw');
%     X1(isnan(X1)) = 0;
%     X2(isnan(X2)) = 0;
%     
%     if i*blocksize < nS
%       columns = 1+(i-1)*blocksize:i*blocksize;    
%     else
%       columns = 1+(i-1)*blocksize:nS;
%     end
%     
%     % % computing R12 and R11 blocks
%     
%     A  = X1'*X2(:,columns); % R_12
%     A_ = X1'*X1(:,columns); % R_11
%     
%     B1 = X1'*M2(:,columns);
%     B1_= X1'*M1(:,columns);
%     
%     B2 = M1'*X2(:,columns);
%     B2_= M1'*X1(:,columns);
%     
%     clear 'X1';
%     clear 'X2';
%     
%     S1 = (T').^2;
%     S2 = (T_tw').^2;
%     S1 = full(S1);
%     S2 = full(S2);
%     
%     C  = M1'*M2(:,columns);
%     
%     N  = A - (B1.*B2)./C;
%     clear 'A';
%     
%     D1 = (S1'*M2(:,columns))-(B1.^2)./C;
%     clear 'M2' 'B1'; 
%     D2 = (M1'*S2(:,columns))-(B2.^2)./C;
%     clear 'S2' 'B2'; 
%     clear 'C';
%     D  = sqrt(D1.*D2);
%     clear 'D1' 'D2';
%     R12 = N./D;
%     clear 'N' 'D';
%     R12(isnan(R12)) = 0;  % AHZ
%     R12(isinf(R12)) = 0;  % AHZ
% 
%     C_  = M1'*M1(:,columns);
%     N_  = A_ - (B1_.*B2_)./C_; 
%     clear 'A_';
%     
%     D1_ = (S1'*M1(:,columns))-(B1_.^2)./C_;
%     clear 'B1_';
%     D2_ = (M1'*S1(:,columns))-(B2_.^2)./C_;
%     clear 'S1' 'M1' 'B2_' 'C_';                                             
%     D_  = sqrt(D1_.*D2_);
%     clear 'D1_' 'D2_';
%     R11 = N_./D_;
%     clear 'N_' 'D_';
%     R11(isnan(R11)) = 0;  % AHZ
%     R11(isinf(R11)) = 0;  % AHZ
%     
%     % % computing AminusB & AplusB
%     AmB = single(R11 - R12); 
%     ApB = single(R11 + R12);
%     clear 'R11' 'R12';
%     
%     save([filePath_AmB,'AmB_block',int2str(i),'.mat'],'AmB','-v7.3');
%     save([filePath_ApB,'ApB_block',int2str(i),'.mat'],'ApB','-v7.3');
%     clear 'AmB' 'ApB';
% end
% toc
% clearvars -except blocksize nS fileTWN % to preserve memory
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % merging AminusB files and performing "eigs" on that:
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% filePath_AmB = './AmB/';
% listing = dir('./AmB');
% numDataFile = length(listing) - 2;  % ignore first two listings: ". & .."
% 
% %nS = 152677;
% %nS = 5000;   % TEST data
% %nS = 120000;
% AminusB = zeros(nS, nS);
% 
% for fileID = 1:numDataFile
%  tic   
%  fileData =[filePath_AmB,'AmB_block',int2str(fileID),'.mat'];
%  load( fileData, 'AmB' );
%  toc
%     if fileID*blocksize < nS
%       columns = 1+(fileID-1)*blocksize:fileID*blocksize;    
%     else
%       columns = 1+(fileID-1)*blocksize:nS;
%     end
%  AmB(imag(AmB)~=0)=0;   %#ok<SAGROW> %AHZ, July-11, to avoid complex numbers
%  AminusB(:,columns) = double(AmB); 
%  disp([datestr(now) sprintf(',  fileID#%d done.',fileID)])
%  clear AmB
% end
% clearvars -except blocksize AminusB nS fileTWN % to preserve memory
% 
% % % computing eigenvalues and eigenvectors:
% [Vasym, Lasym, flag] = eigs(AminusB,6);                                  
% toc
% save V_L_AminusB.mat Vasym Lasym
% clearvars -except blocksize nS fileTWN% to preserve memory
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % merging AplusB files and performing "eigs" on that:
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% tic
% filePath_ApB = './ApB/';
% listing = dir('./ApB');
% numDataFile = length(listing) - 2;  % ignore first two listings: ". & .."
% 
% %nS = 152677;  % the original number of snapshots
% %disp('make sure about nS here, which is the original num of data!');
% %nS = 5000;   % TEST data
% %nS = 120000;
% AplusB = zeros(nS, nS);
% 
% % %blocksize = 1000;
% for fileID = 1:numDataFile
%  tic   
%  fileData =[filePath_ApB,'ApB_block',int2str(fileID),'.mat'];
%  load( fileData, 'ApB' );
%  toc
%     if fileID*blocksize < nS
%       columns = 1+(fileID-1)*blocksize:fileID*blocksize;    
%     else
%       columns = 1+(fileID-1)*blocksize:nS;
%     end
%  ApB(imag(ApB)~=0)=0;  %AHZ, July-11, to avoid complex numbers
%  AplusB(:,columns) = double(ApB); 
%  disp([datestr(now) sprintf(',  fileID#%d done.',fileID)])
% end
% clearvars -except AplusB fileTWN
% 
% % computing eigenvalues and eigenvectors:
% tic
% [Vsym, Lsym, flag] = eigs(AplusB,6);
% toc
% save V_L_AplusB.mat Vsym Lsym
% clear

% % %&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&  STEP(3)  &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&  
% % % % Finding disputed snapshots via "A-B" approach
% 
load V_L_AminusB.mat Vasym Lasym
load V_L_AplusB.mat Vsym Lsym

%save PYP_femto_detwin_eigVecs_eigVals_DRL_SCL_nS152677.mat Lasym Lsym Vasym Vsym

[dummy, lind] = sort(diag(Lasym),'descend'); %#ok<ASGLU> % sort according to eigen values
Lasym = Lasym(lind,lind);
Vasym = Vasym(:,lind);

%%%%%%%%%%%%%%%%%%%
figure; bar(diag(Lasym))
saveas(gcf, 'Eigenvalues_AminusB.png');
close(gcf);

Vasym_full = [Vasym(:,1);-Vasym(:,1)];

figure; hist(Vasym_full,200); 
saveas(gcf, 'First_EVector_AminusB_mirrored.png');
close(gcf);
%xlim([-0.001 0.001])
%%%%%%%%%%%%%%%%%%%

[dummy, lind] = sort(diag(Lsym),'descend'); % sort according to eigen values
Lsym = Lsym(lind,lind);
Vsym = Vsym(:,lind);

%%%%%%%%%%%%%%%%%%%
figure; bar(diag(Lsym))
saveas(gcf, 'Eigenvalues_AplusB.png');
close(gcf);

Vsym_full = [Vsym(:,1);Vsym(:,1)];

figure; hist(Vsym_full,200); 
saveas(gcf, 'First_EVector_AplusB_concat.png');
close(gcf);
%xlim([-0.001 0.001])
%%%%%%%%%%%%%%%%%%%%%%

% the full symmetric
nS = size(Vasym,1)                                                         %#ok<NOPTS>
figure
scatter(Vasym_full(1:nS,1),Vsym_full(1:nS,1),4,'b'); 
box on; 
axis image;
hold on;
scatter(Vasym_full(nS+1:2*nS,1),Vsym_full(nS+1:2*nS,1),4,'r'); 
axis image;
xlabel('Eigenvector #1 (asym, full)'); 
ylabel('Eigenvector #1 (sym, full)');
saveas(gcf, 'EV1_AminusB_mirrored_EV1_AplusB_concat.png');
close(gcf);
% to check if the data is already detwinned 
% figure;
% scatter(Vasym(:,1),Vsym(:,1),4,'b'); box on; axis image
% xlabel('Eigenvector #1 of "A-B"'); ylabel('Eigenvector #1 of "A+B"');

ind_dispute = find(Vasym(:,1)<0); 
length(ind_dispute)

%%%%%%%
figure;
Vasym_ = Vasym(:,1);
Vsym_  = Vsym(:,1);
Vasym_(ind_dispute)=[];
Vsym_(ind_dispute) =[];
scatter(Vasym_,Vsym_,4,'b'); 
box on; 
axis image;
xlabel('Eigenvector #1 of "A-B"'); ylabel('Eigenvector #1 of "A+B"');
title('disputed snapshots removed.');
saveas(gcf, 'EV1_AminusB_EV1_AplusB_disputed_snapshots_removed.png');
close(gcf);

clarify_disputed_snapshots = true;

if clarify_disputed_snapshots
    % % Option (1): calrification; removing disputed snapshots:

    load(fileTWN); % data_bR_light_int_unifdelay_DRL_SCL_nS127028_nBrg54391_twinned.mat

    T_dtw = T;
    M_dtw = M;
    T_dtw(ind_dispute,:) = []; 
    M_dtw(ind_dispute,:) = []; 
    %CMC - ONLY LIGHT DATA 
    %t_uniform(ind_dispute) = [];
    
    [nS, nBrg] = size(T_dtw);

    myfileData = sprintf('data_bR_dark_no_laser_int_DRL_SCL_DTW_nS%d_nBrg%d.mat',nS,nBrg);
    save(myfileData,'T_dtw',...
                    'M_dtw',...
                    'miller_h',...
                    'miller_k',...
                    'miller_l',...
                    'ind_dispute',...
                    '-v7.3');
    %CMC - And t_uniform for light data
    clear

elseif detwin_disputed_snapshots
    % % Option (2): detwinning the data without removing disputed snapshots

    load dataPYP_femto_int_sortdelay_unifdelay_DRL_SCL_nS152677_nBrg21556_twinned.mat

    X1 = full(T');
    X2 = full(T_tw');
    M1 = full(M');
    M2 = full(M_tw');

    IND = find(Vasym_full(:,1)>=0);
    length(IND)

    figure;
    scatter(Vasym_full(IND,1),Vsym_full(IND,1),4,'b'); box on; axis image
    xlabel('Eigenvector #1 (asym, full)'); ylabel('Eigenvector #1 (sym, full)');
    title(['detwinned, nS=',int2str(152,503)])


    XX = [X1,X2];
    Xdetw = XX(:,IND);
    size(Xdetw)
    MM = [M1,M2];
    Mdetw = MM(:,IND);
    size(Mdetw)

    T_dtw = sparse(Xdetw'); 
    M_dtw = sparse(Mdetw');
    ind_dtw = IND;
    notice='disputes snapshots NOT removed but detwinned';
    [nS, nBrg] = size(T_dtw)

    myfileTWN = sprintf('dataPYP_femto_int_sortdelay_unifdelay_DRL_SCL_DTW_nS%d_nBrg%d.mat',nS,nBrg);

    save(myfileTWN,'T_dtw','M_dtw','ind_dtw','miller_h','miller_k','miller_l','delay',...
                 'notice_negative_pix','notice','sort_notice','-v7.3');

    clear

else
    
    disp('no option considered for detwinning!');
    
end

%
%EOF
