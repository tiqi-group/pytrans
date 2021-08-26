clear all;

global q m ycent
q = 1.6e-19; 
m = 40*1.67e-27; 
Omega = 2*pi*34e6; 


nptsx = 500;
nptsy = 5000;

xrange = 6; 
yrange = 6; 
ycent = 51.5; 
Vstot = zeros(nptsy, nptsx); 

%import Axial confinement data
currname = 'VAxialConf.csv'; 
[VmAx, ym, zm] = unpack2Ddata(currname, nptsx, nptsy);
[VmbAx, ymb, zmb] = bound2Ddata(VmAx, ym, zm, xrange, yrange, ycent); 

figure(1); subplot(1,3,1); surf(ymb,zmb, VmbAx);
shading flat; colormap hot; view([0 90]);
axis equal
colorbar 
xlim([-xrange xrange])
ylim([ycent-yrange ycent+yrange])

%import Tilt data
currname = 'VTilt.csv'; 
[VmTilt, ym, zm] = unpack2Ddata(currname, nptsx, nptsy);
[VmbTilt, ymb, zmb] = bound2Ddata(VmTilt, ym, zm, xrange, yrange, ycent); 

figure(1); subplot(1,3,2); surf(ymb,zmb, VmbTilt);
shading interp; colormap hot; view([0 90]);
axis equal
colorbar 
xlim([-xrange xrange])
ylim([ycent-yrange ycent+yrange])

%import Pseudopotential data
currname = 'normE_RF.csv'; 
[normErfm, ym, zm] = unpack2Ddata(currname, nptsx, nptsy);
[normErfmb, ymb, zmb] = bound2Ddata(normErfm, ym, zm, xrange, yrange, ycent); 

figure(1); subplot(1,3,3); surf(ymb,zmb, normErfmb);
shading interp; colormap hot; view([0 90]);
axis equal
colorbar 
xlim([-xrange xrange])
ylim([ycent-yrange ycent+yrange])


%% Calculate potentials for given operating parameters

Vrfs = linspace(30,40,50); 
Vrfs = 34
radfreqs = zeros(2, length(Vrfs)); 
radthetas = zeros(1, length(Vrfs)); 
drawtheplots = true; 
for k=1:length(Vrfs)
Vrf = Vrfs(k); %RF amplitude in Volts
Ax = 1.2; %MHz, axial frequency; Axial votlage above is for 1 MHz
Tilt = 0*-9; %Current value of tilt -- simply multiplies the set above as currently implemented

Um = (Vrf*normErfmb).^2*q^2/2/m/Omega^2/q/2; %Pseudopotential in eV, with E2 corresponding to the peak E field squared
if(drawtheplots)
    figure; surf(ymb, zmb, Um); 
shading interp; colormap hot; view([0 90]);
axis equal
colorbar 
xlim([-xrange xrange])
ylim([ycent-yrange ycent+yrange])
end 

TotPot = Um + Ax.^2*VmbAx + Tilt*VmbTilt; 
if(drawtheplots)
    figure; subplot(1,2,1); surf(ymb, zmb, TotPot); 
shading interp; colormap hot; view([0 90]);
axis equal
colorbar 
xlim([-xrange xrange])
ylim([ycent-yrange ycent+yrange])
title('Total Potential Simulated')
xlabel('y (microns)')
ylabel('z (microns)')
end 

% Fit with a 2D function
XY(:,:,1) = ymb; XY(:,:,2) = zmb; 


options = optimoptions('lsqcurvefit', 'FunctionTolerance', 1e-15, ...
    'OptimalityTolerance', 1e-12, ...
    'StepTolerance', 1e-9, ...
    'Display', 'iter'); 
% optioins = optimoptions('lsqcurvefit'); 
params0 = [pi/8  2*pi*4   2*pi*5   -.444];

fitparams = lsqcurvefit(@TiltedPotential, params0, XY, TotPot, [], [], options); 

if(drawtheplots)
    subplot(1,2,2); surf(ymb, zmb, TiltedPotential(fitparams, XY))
shading interp; colormap hot; view([0 90]);
axis equal
colorbar 
xlim([-xrange xrange])
ylim([ycent-yrange ycent+yrange])
title('Total Potential Fitted')
xlabel('y (microns)')
ylabel('z (microns)')
end

radfs = [fitparams(2); fitparams(3)]/2/pi; 
radt = mod(fitparams(1), 2*pi)*180/pi;
radfreqs(:,k) = radfs; 
radthetas(k) = radt; 
disp('Radial trap frequencies in MHz:')
disp(radfs)
disp('Radial mode tilt')
disp(radt)
k
end

figure;  subplot(1,2,1); plot(Vrfs', radfreqs'); 
xlabel('RF peak amplitude (V)')
ylabel('Frequency (MHz)')
legend('mode 1', 'mode 2')
subplot(1,2,2); plot(Vrfs, radthetas); 
legend(['mode 1'])
xlabel('RF peak amplitude (V)')
ylabel('Mode 1 angle to horizontal (degrees)')





function[Vmb, ymb, zmb] = bound2Ddata(Vm, ym, zm, xrange, yrange, ycent)
    ymin = ycent-yrange;
    ymax = ycent+yrange;
    xmin = -xrange; 
    xmax = xrange; 
    yinds = find((ym(1,:)>xmin).*(ym(1,:)<xmax));
    zinds = find((zm(:,1)>ymin).*(zm(:,1)<ymax));

    ymb = ym(zinds, yinds);
    zmb = zm(zinds, yinds); 
    Vmb = Vm(zinds, yinds); 
end

function [Vm, ym, zm] = unpack2Ddata(filename, nptsx, nptsy)
data = csvread(filename, 9);

x = data(1:nptsx,1);
y = data(1:nptsx:end,2);
V = data(:,3); 
Vm = zeros(nptsx, nptsy);
for k = 1:nptsy
    ind0 = 1+(k-1)*nptsx;
    Vm(:,k) = V(ind0:(ind0+nptsx-1));
end
[xmraw, ymraw] = meshgrid(x,y);
ym = ymraw';
zm = xmraw';
end