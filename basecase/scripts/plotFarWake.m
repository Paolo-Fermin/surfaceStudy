clear all

% set some simulation info
dataStart = 0;
% change the dataEnd to 3600
dataEnd =120;
dataInterval = 30;
Uship = 5.0
cellX = Uship*dataInterval;
path = '../postProcessing/sample/';

% read in the first velocity data
fid = fopen([path '0/surface_U.xy']);
fileDataU = textscan(fid,'%f%f%f%f');
fclose(fid);

% now we know the size of the arrays
NYpoints = length(fileDataU{1});
NXpoints = (dataEnd/dataInterval)+1;

x = zeros(NXpoints,NYpoints);
y = zeros(NXpoints,NYpoints);
Ux = zeros(NXpoints,NYpoints);
Uy = zeros(NXpoints,NYpoints);
Uz = zeros(NXpoints,NYpoints);

% we loop over each time and fill in x and y
for n=1:NXpoints
    
   
    % now fill in the surface data for U
    fidU = fopen([path int2str((n-1)*dataInterval) '/surface_U.xy']);
    fileDataU = textscan(fidU,'%f%f%f%f');
    fclose(fidU);
    x(n,:) = (n-1)*cellX;
    y(n,:) = fileDataU{1};
    Ux(n,:) = fileDataU{2};
    Uy(n,:) = fileDataU{3};
    Uz(n,:) = fileDataU{4};

    
end

xSym = [x(1:end,1:(end-1)),x(1:end,:)]; 
ySym = [-y(:,end:-1:2), y(:,1:end)];
UxSym = [-Ux(:,end:-1:2), Ux(:,1:end)];
UySym = [-Uy(:,end:-1:2), Uy(:,1:end)];
UzSym = [-Uz(:,end:-1:2), Uz(:,1:end)];

UxSym(abs(UxSym)<1.0E-10)=0;
UySym(abs(UySym)<1.0E-10)=0;
UzSym(abs(UzSym)<1.0E-10)=0;

% plot the transverse velocity
%figure(1)
%clf
%hold on

%hfig = pcolor(xSym,ySym,UxSym);
%set(hfig,'edgeColor','none','FaceColor','interp')

%cmin=-0.00001;
%cmax=0.00001;
%caxis([cmin cmax])
%hbar = colorbar;

%box on
%xlabel('Axial Distance [m]','FontSize', 14)
%ylabel('Transverse Distance [m]','FontSize', 14)
%set(gca,'FontSize',14)
%title('Axial (Along-Track) Surface Velocity [m/s]','FontSize',14)

%csvwrite('UxSym.csv',UxSym);


% plot the transverse velocity
%figure(2)
%clf
%hold on

%hfig = pcolor(xSym,ySym,UySym);
%set(hfig,'edgeColor','none','FaceColor','interp')

%cmin=-0.00001;
%cmax=0.00001;
%caxis([cmin cmax])
%hbar = colorbar;

%box on
%xlabel('Axial Distance [m]','FontSize', 14)
%ylabel('Transverse Distance [m]','FontSize', 14)
%set(gca,'FontSize',14)
%title('Transverse (Off-Track) Surface Velocity [m/s]','FontSize',14)

csvwrite('UySym.csv',UySym);

% plot the transverse velocity
%figure(3)
%clf
%hold on

%hfig = pcolor(xSym,ySym,UzSym);
%set(hfig,'edgeColor','none','FaceColor','interp')

%cmin=-0.00001;
%cmax=0.00001;
%caxis([cmin cmax])
%hbar = colorbar;

%box on
%xlabel('Axial Distance [m]','FontSize', 14)
%ylabel('Transverse Distance [m]','FontSize', 14)
%set(gca,'FontSize',14)
%title('Vertical Surface Velocity [m/s]','FontSize',14)

csvwrite('UzSym.csv',UzSym);
