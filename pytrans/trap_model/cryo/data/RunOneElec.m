function out = RunOneElec(elecname, elecdoms, x0, sflag)
%
% Axial2qubitTrapSimDC.m
%
% Model exported on Dec 5 2018, 14:02 by COMSOL 5.3.0.260.

import com.comsol.model.*
import com.comsol.model.util.*

model = ModelUtil.create('Model');

model.modelPath('/scratch/TrapSimulation_Karan');

model.label('2018_12_05_Axial2qubitTrapSim_DC.mph');

model.comments(['Untitled\n\n']);

model.param.set('domh', '1.5[mm]');
model.param.set('domxmin', '-3 [mm]');
model.param.set('domxmax', '5 [mm]');
model.param.set('domymin', '-1.8 [mm]');
model.param.set('domymax', '1.8 [mm]');
model.param.set('q', '1.6e-19[C]');
model.param.set('m', '40*1.67e-27 [kg]');
model.param.set('Omega', '2*pi*30e6 [Hz]');
model.param.set('boxt', '100  [um]');
model.param.set('traph', '51.5 [um]');

model.component.create('comp1', true);

model.component('comp1').geom.create('geom1', 3);

model.result.table.create('evl2', 'Table');

model.component('comp1').mesh.create('mesh1');

model.component('comp1').geom('geom1').lengthUnit([native2unicode(hex2dec({'00' 'b5'}), 'unicode') 'm']);
model.component('comp1').geom('geom1').create('blk1', 'Block');
model.component('comp1').geom('geom1').feature('blk1').label('SimulationDomain');
model.component('comp1').geom('geom1').feature('blk1').set('pos', {'domxmin' 'domymin' '-boxt'});
model.component('comp1').geom('geom1').feature('blk1').set('size', {'domxmax-domxmin' 'domymax-domymin' 'domh-(domh/2+boxt)'});
model.component('comp1').geom('geom1').create('blk2', 'Block');
model.component('comp1').geom('geom1').feature('blk2').label('BottomOxide');
model.component('comp1').geom('geom1').feature('blk2').set('pos', {'domxmin' 'domymin' '-boxt'});
model.component('comp1').geom('geom1').feature('blk2').set('size', {'domxmax-domxmin' 'domymax-domymin' 'boxt'});
model.component('comp1').geom('geom1').create('wp1', 'WorkPlane');
model.component('comp1').geom('geom1').feature('wp1').active(false);
model.component('comp1').geom('geom1').feature('wp1').set('quickplane', 'xz');
model.component('comp1').geom('geom1').feature('wp1').set('unite', true);
model.component('comp1').geom('geom1').create('blk3', 'Block');
model.component('comp1').geom('geom1').feature('blk3').label('MeshBox');
model.component('comp1').geom('geom1').feature('blk3').set('pos', [0 0 30]);
model.component('comp1').geom('geom1').feature('blk3').set('base', 'center');
model.component('comp1').geom('geom1').feature('blk3').set('size', [500 100 80]);
model.component('comp1').geom('geom1').create('imp1', 'Import');
model.component('comp1').geom('geom1').feature('imp1').set('selresult', true);
model.component('comp1').geom('geom1').feature('imp1').set('selresultshow', 'all');
model.component('comp1').geom('geom1').feature('imp1').set('type', 'gds');
model.component('comp1').geom('geom1').feature('imp1').set('filename', '/scratch/TrapSimulation_Karan/2018_12_05_Axial_2qubit_trap.gds');
model.component('comp1').geom('geom1').feature('imp1').set('importtype', 'shell');
model.component('comp1').geom('geom1').feature('imp1').set('manualelevation', true);
model.component('comp1').geom('geom1').feature('imp1').set('layername', {'LAYER10' 'LAYER20' 'LAYER30' 'LAYER35' 'LAYER36' 'LAYER37'});
model.component('comp1').geom('geom1').feature('imp1').set('height', {'0' '0' '0' '.2' '0' '.3'});
model.component('comp1').geom('geom1').feature('imp1').set('elevation', [0 0 0 -3 0 0]);
model.component('comp1').geom('geom1').feature('imp1').set('importlayer', {'off' 'off' 'off' 'on' 'off' 'on'});
model.component('comp1').geom('geom1').feature('imp1').set('repairgeom', false);
model.component('comp1').geom('geom1').feature('imp1').importData;
model.component('comp1').geom('geom1').create('blk4', 'Block');
model.component('comp1').geom('geom1').feature('blk4').label('MeshBox 1');
model.component('comp1').geom('geom1').feature('blk4').set('pos', [0 0 95]);
model.component('comp1').geom('geom1').feature('blk4').set('base', 'center');
model.component('comp1').geom('geom1').feature('blk4').set('size', [1500 500 200]);
model.component('comp1').geom('geom1').run;
model.component('comp1').geom('geom1').run('fin');

model.component('comp1').selection.create('sel1', 'Explicit');
model.component('comp1').selection('sel1').label('MeshBox');

model.view.create('view3', 2);
model.view.create('view4', 2);
model.view.create('view5', 2);
model.view.create('view6', 2);
model.view.create('view7', 2);

model.component('comp1').material.create('mat1', 'Common');
model.component('comp1').material.create('mat4', 'Common');
model.component('comp1').material.create('mat5', 'Common');
model.component('comp1').material.create('mat3', 'Common');
model.component('comp1').material('mat1').selection.set([1 3 5 6]);
model.component('comp1').material('mat4').selection.named('geom1_imp1_dom');
model.component('comp1').material('mat3').selection.set([2 4 7]);

model.component('comp1').physics.create('ec', 'ConductiveMedia', 'geom1');
model.component('comp1').physics('ec').create('pot2', 'ElectricPotential', 2);
model.component('comp1').physics('ec').feature('pot2').selection.named('geom1_imp1_LAYER35_bnd');
model.component('comp1').physics('ec').create('pot3', 'ElectricPotential', 2);
model.component('comp1').physics('ec').feature('pot3').selection.named('geom1_imp1_LAYER37_bnd');
model.component('comp1').physics('ec').create('pot1', 'ElectricPotential', 2);
model.component('comp1').physics('ec').feature('pot1').selection.set(elecdoms); % THESE DOMAINS GET 1 V APPLIED

model.component('comp1').mesh('mesh1').create('ftet1', 'FreeTet');
model.component('comp1').mesh('mesh1').feature('ftet1').create('size1', 'Size');
model.component('comp1').mesh('mesh1').feature('ftet1').create('size2', 'Size');
model.component('comp1').mesh('mesh1').feature('ftet1').feature('size1').selection.geom('geom1', 3);
model.component('comp1').mesh('mesh1').feature('ftet1').feature('size1').selection.set([7]);
model.component('comp1').mesh('mesh1').feature('ftet1').feature('size2').selection.geom('geom1', 3);
model.component('comp1').mesh('mesh1').feature('ftet1').feature('size2').selection.set([4]);

model.result.table('evl2').label('Evaluation 2D');
model.result.table('evl2').comments('Interactive 2D values');

model.component('comp1').view('view1').set('renderwireframe', true);
model.component('comp1').view('view1').set('scenelight', false);
model.component('comp1').view('view1').set('transparency', true);
model.component('comp1').view('view2').axis.set('xmin', -3400);
model.component('comp1').view('view2').axis.set('xmax', 5400);
model.component('comp1').view('view2').axis.set('ymin', -2779.53955078125);
model.component('comp1').view('view2').axis.set('ymax', 2779.53955078125);
model.component('comp1').view('view2').axis.set('abstractviewlratio', -0.05000000074505806);
model.component('comp1').view('view2').axis.set('abstractviewrratio', 0.05000000074505806);
model.component('comp1').view('view2').axis.set('abstractviewbratio', -0.4265131950378418);
model.component('comp1').view('view2').axis.set('abstractviewtratio', 0.4265131950378418);
model.component('comp1').view('view2').axis.set('abstractviewxscale', 11.253196716308594);
model.component('comp1').view('view2').axis.set('abstractviewyscale', 11.253196716308594);
model.view('view3').axis.set('xmin', -3250.08740234375);
model.view('view3').axis.set('xmax', 2800.08740234375);
model.view('view3').axis.set('ymin', -1980);
model.view('view3').axis.set('ymax', 1980);
model.view('view3').axis.set('abstractviewlratio', -4.153980731964111);
model.view('view3').axis.set('abstractviewrratio', 4.153980731964111);
model.view('view3').axis.set('abstractviewbratio', -0.05000000074505806);
model.view('view3').axis.set('abstractviewtratio', 0.05000000074505806);
model.view('view3').axis.set('abstractviewxscale', 6.2957072257995605);
model.view('view3').axis.set('abstractviewyscale', 6.295707702636719);
model.view('view4').axis.set('xmin', -1397.6845703125);
model.view('view4').axis.set('xmax', 1263.26220703125);
model.view('view4').axis.set('ymin', -627.2870483398438);
model.view('view4').axis.set('ymax', 1114.3731689453125);
model.view('view4').axis.set('abstractviewlratio', 0.45707598328590393);
model.view('view4').axis.set('abstractviewrratio', -0.21441207826137543);
model.view('view4').axis.set('abstractviewbratio', -0.8112109303474426);
model.view('view4').axis.set('abstractviewtratio', 0.8682665824890137);
model.view('view4').axis.set('abstractviewxscale', 2.768935203552246);
model.view('view4').axis.set('abstractviewyscale', 2.768935441970825);
model.view('view5').axis.set('xmin', -4085.70166015625);
model.view('view5').axis.set('xmax', 6085.70166015625);
model.view('view5').axis.set('ymin', -2691.41259765625);
model.view('view5').axis.set('ymax', 4191.41259765625);
model.view('view5').axis.set('abstractviewlratio', -0.05000000074505806);
model.view('view5').axis.set('abstractviewrratio', 0.05000000074505806);
model.view('view5').axis.set('abstractviewbratio', -1.452737808227539);
model.view('view5').axis.set('abstractviewtratio', 1.452737808227539);
model.view('view5').axis.set('abstractviewxscale', 12.680115699768066);
model.view('view5').axis.set('abstractviewyscale', 12.680115699768066);
model.view('view6').axis.set('xmin', -3250.08740234375);
model.view('view6').axis.set('xmax', 2800.08740234375);
model.view('view6').axis.set('ymin', -1980);
model.view('view6').axis.set('ymax', 1980);
model.view('view6').axis.set('abstractviewlratio', -4.153980731964111);
model.view('view6').axis.set('abstractviewrratio', 4.153980731964111);
model.view('view6').axis.set('abstractviewbratio', -0.05000000074505806);
model.view('view6').axis.set('abstractviewtratio', 0.05000000074505806);
model.view('view6').axis.set('abstractviewxscale', 6.2957072257995605);
model.view('view6').axis.set('abstractviewyscale', 6.295707702636719);
model.view('view7').axis.set('xmin', -6722.41650390625);
model.view('view7').axis.set('xmax', 6722.41650390625);
model.view('view7').axis.set('ymin', -3400);
model.view('view7').axis.set('ymax', 5400);
model.view('view7').axis.set('abstractviewlratio', -1.367337942123413);
model.view('view7').axis.set('abstractviewrratio', 1.367337942123413);
model.view('view7').axis.set('abstractviewbratio', -0.05000000074505806);
model.view('view7').axis.set('abstractviewtratio', 0.05000000074505806);
model.view('view7').axis.set('abstractviewxscale', 13.990461349487305);
model.view('view7').axis.set('abstractviewyscale', 13.990461349487305);

model.component('comp1').material('mat1').label('Oxide');
model.component('comp1').material('mat1').propertyGroup('def').set('electricconductivity', {'0' '0' '0' '0' '0' '0' '0' '0' '0'});
model.component('comp1').material('mat1').propertyGroup('def').set('relpermittivity', {'3.8' '0' '0' '0' '3.8' '0' '0' '0' '3.8'});
model.component('comp1').material('mat4').label('Gold');
model.component('comp1').material('mat4').propertyGroup('def').set('electricconductivity', {'10^8' '0' '0' '0' '10^8' '0' '0' '0' '10^8'});
model.component('comp1').material('mat4').propertyGroup('def').set('relpermittivity', {'1' '0' '0' '0' '1' '0' '0' '0' '1'});
model.component('comp1').material('mat5').label('Platinum');
model.component('comp1').material('mat5').propertyGroup('def').set('electricconductivity', {'10^8' '0' '0' '0' '10^8' '0' '0' '0' '10^8'});
model.component('comp1').material('mat5').propertyGroup('def').set('relpermittivity', {'1' '0' '0' '0' '1' '0' '0' '0' '1'});
model.component('comp1').material('mat3').label('Air');
model.component('comp1').material('mat3').propertyGroup('def').set('electricconductivity', {'0' '0' '0' '0' '0' '0' '0' '0' '0'});
model.component('comp1').material('mat3').propertyGroup('def').set('relpermittivity', {'1' '0' '0' '0' '1' '0' '0' '0' '1'});

model.component('comp1').physics('ec').feature('pot1').set('V0', 1);

model.component('comp1').mesh('mesh1').feature('size').set('hauto', 2);
model.component('comp1').mesh('mesh1').feature('size').set('custom', 'on');
model.component('comp1').mesh('mesh1').feature('size').set('hmax', 280);
model.component('comp1').mesh('mesh1').feature('size').set('hmin', 12);
model.component('comp1').mesh('mesh1').feature('size').set('hgrad', 1.2);
model.component('comp1').mesh('mesh1').feature('ftet1').feature('size1').set('custom', 'on');
model.component('comp1').mesh('mesh1').feature('ftet1').feature('size1').set('hmax', 4);
model.component('comp1').mesh('mesh1').feature('ftet1').feature('size1').set('hmaxactive', true);
model.component('comp1').mesh('mesh1').feature('ftet1').feature('size1').set('hmin', 1);
model.component('comp1').mesh('mesh1').feature('ftet1').feature('size1').set('hminactive', false);
model.component('comp1').mesh('mesh1').feature('ftet1').feature('size2').set('custom', 'on');
model.component('comp1').mesh('mesh1').feature('ftet1').feature('size2').set('hmax', 10);
model.component('comp1').mesh('mesh1').feature('ftet1').feature('size2').set('hmaxactive', true);
model.component('comp1').mesh('mesh1').feature('ftet1').feature('size2').set('hmin', 1);
model.component('comp1').mesh('mesh1').feature('ftet1').feature('size2').set('hminactive', false);
model.component('comp1').mesh('mesh1').run;

model.study.create('std1');
model.study('std1').create('freq', 'Frequency');

model.sol.create('sol1');
model.sol('sol1').study('std1');
model.sol('sol1').attach('std1');
model.sol('sol1').create('st1', 'StudyStep');
model.sol('sol1').create('v1', 'Variables');
model.sol('sol1').create('s1', 'Stationary');
model.sol('sol1').feature('s1').create('p1', 'Parametric');
model.sol('sol1').feature('s1').create('fc1', 'FullyCoupled');
model.sol('sol1').feature('s1').create('i1', 'Iterative');
model.sol('sol1').feature('s1').feature('i1').create('mg1', 'Multigrid');
model.sol('sol1').feature('s1').feature.remove('fcDef');

model.result.dataset.create('dset2', 'Solution');
model.result.dataset.create('cpl3', 'CutPlane');
model.result.dataset.create('cpl1', 'CutPlane');
model.result.dataset.create('cpl2', 'CutPlane');
model.result.dataset.create('cln1', 'CutLine3D');
model.result.dataset.create('cln2', 'CutLine3D');
model.result.dataset.create('cln3', 'CutLine3D');
model.result.dataset.create('cln4', 'CutLine3D');
model.result.dataset.create('cln5', 'CutLine3D');

model.result.create('pg1', 'PlotGroup3D');
model.result.create('pg2', 'PlotGroup2D');
model.result.create('pg3', 'PlotGroup2D');
model.result.create('pg4', 'PlotGroup2D');
model.result.create('pg5', 'PlotGroup1D');
model.result.create('pg6', 'PlotGroup1D');
model.result.create('pg7', 'PlotGroup1D');
model.result('pg1').create('mslc1', 'Multislice');
model.result('pg1').create('surf1', 'Surface');
model.result('pg2').create('surf1', 'Surface');
model.result('pg3').create('surf1', 'Surface');
model.result('pg4').create('surf1', 'Surface');
model.result('pg5').create('lngr1', 'LineGraph');
model.result('pg6').create('lngr1', 'LineGraph');
model.result('pg7').create('lngr1', 'LineGraph');
model.result.export.create('data1', 'Data');
model.result.export.create('data2', 'Data');
model.result.export.create('data3', 'Data');
model.result.export.create('data4', 'Data');
model.result.export.create('data5', 'Data');
model.result.export.create('data6', 'Data');
model.result.export.create('data7', 'Data');
model.result.export.create('data8', 'Data');
model.result.export.create('data9', 'Data');
model.result.export.create('data10', 'Data');
model.result.export.create('data11', 'Data');

model.study('std1').feature('freq').setIndex('plist', '1', 0);

model.sol('sol1').attach('std1');
model.sol('sol1').feature('v1').set('clistctrl', {'p1'});
model.sol('sol1').feature('v1').set('cname', {'freq'});
model.sol('sol1').feature('v1').set('clist', {'1[Hz]'});
model.sol('sol1').feature('s1').feature('p1').set('pname', {'freq'});
model.sol('sol1').feature('s1').feature('p1').set('plistarr', [1]);
model.sol('sol1').feature('s1').feature('p1').set('punit', {'Hz'});
model.sol('sol1').feature('s1').feature('p1').set('pcontinuationmode', 'no');
model.sol('sol1').feature('s1').feature('p1').set('preusesol', 'auto');
model.sol('sol1').feature('s1').feature('i1').set('linsolver', 'bicgstab');
model.sol('sol1').runAll;

' Done running' 
model.result.dataset('cpl3').label('Cut Plane x');
model.result.dataset('cpl3').set('planetype', 'general');
model.result.dataset('cpl3').set('genmethod', 'pointnormal');
model.result.dataset('cpl3').set('genpnvec', [1 0 0]);
model.result.dataset('cpl1').label('Cut Plane y');
model.result.dataset('cpl1').set('planetype', 'general');
model.result.dataset('cpl1').set('genmethod', 'pointnormal');
model.result.dataset('cpl1').set('genpnvec', [0 1 0]);
model.result.dataset('cpl2').label('Cut Plane z');
model.result.dataset('cpl2').set('planetype', 'general');
model.result.dataset('cpl2').set('genmethod', 'pointnormal');
model.result.dataset('cpl2').set('genpnpoint', [0 0 51.5]);

model.result.dataset('cln1').label('Cut Line 3D x');
model.result.dataset('cln1').set('method', 'pointdir');
model.result.dataset('cln1').set('pdpoint', [x0 0 51.5]);
model.result.dataset('cln1').set('pddir', [1 0  0]);
model.result.dataset('cln2').label('Cut Line 3D y');
model.result.dataset('cln2').set('method', 'pointdir');
model.result.dataset('cln2').set('pdpoint', [x0 0 51.5]);
model.result.dataset('cln2').set('pddir', [0 1 0]);
model.result.dataset('cln3').label('Cut Line 3D z');
model.result.dataset('cln3').set('method', 'pointdir');
model.result.dataset('cln3').set('pdpoint', [x0 0 51.5]);
model.result.dataset('cln3').set('pddir', [0 0 1]);

model.result.dataset('cln4').label('Cut Line 3D y+z');
model.result.dataset('cln4').set('method', 'pointdir');
model.result.dataset('cln4').set('pdpoint', [x0 0 51.5]);
model.result.dataset('cln4').set('pddir', [0 1 1]);
model.result.dataset('cln5').label('Cut Line 3D y-z');
model.result.dataset('cln5').set('method', 'pointdir');
model.result.dataset('cln5').set('pdpoint', [x0 0 51.5]);
model.result.dataset('cln5').set('pddir', [0 1 -1]);


model.result('pg1').label('Electric Potential (ec)');
model.result('pg1').set('frametype', 'spatial');
model.result('pg1').feature('mslc1').set('expr', 'ec.normE^2');
model.result('pg1').feature('mslc1').set('unit', 'kg^2*m^2/(s^6*A^2)');
model.result('pg1').feature('mslc1').set('descr', 'ec.normE^2');
model.result('pg1').feature('mslc1').set('xnumber', '0');
model.result('pg1').feature('mslc1').set('znumber', '0');
model.result('pg1').feature('mslc1').set('rangecoloractive', true);
model.result('pg1').feature('mslc1').set('rangecolormax', '2e10');
model.result('pg1').feature('mslc1').set('resolution', 'normal');
model.result('pg1').feature('surf1').active(false);
model.result('pg1').feature('surf1').set('resolution', 'normal');
model.result('pg2').label('V_xplane');
model.result('pg2').set('frametype', 'spatial');
model.result('pg2').feature('surf1').set('resolution', 'normal');
model.result('pg3').label('V_yplane');
model.result('pg3').set('data', 'cpl1');
model.result('pg3').set('frametype', 'spatial');
model.result('pg3').feature('surf1').set('resolution', 'normal');
model.result('pg4').label('V_zplane');
model.result('pg4').set('data', 'cpl2');
model.result('pg4').set('frametype', 'spatial');
model.result('pg4').feature('surf1').set('resolution', 'normal');
model.result('pg5').label('V_xline');
model.result('pg5').set('xlabel', 'Arc length');
model.result('pg5').set('ylabel', 'Electric potential (V)');
model.result('pg5').set('xlabelactive', false);
model.result('pg5').set('ylabelactive', false);
model.result('pg5').feature('lngr1').set('data', 'cln1');
model.result('pg5').feature('lngr1').set('looplevelinput', {'all'});
model.result('pg5').feature('lngr1').set('resolution', 'normal');
model.result('pg6').label('V_yline');
model.result('pg6').set('xlabel', 'Arc length');
model.result('pg6').set('ylabel', 'Electric potential (V)');
model.result('pg6').set('xlabelactive', false);
model.result('pg6').set('ylabelactive', false);
model.result('pg6').feature('lngr1').set('data', 'cln2');
model.result('pg6').feature('lngr1').set('looplevelinput', {'all'});
model.result('pg6').feature('lngr1').set('resolution', 'normal');
model.result('pg7').label('V_zline');
model.result('pg7').set('xlabel', 'Arc length');
model.result('pg7').set('ylabel', 'Electric potential (V)');
model.result('pg7').set('xlabelactive', false);
model.result('pg7').set('ylabelactive', false);
model.result('pg7').feature('lngr1').set('data', 'cln3');
model.result('pg7').feature('lngr1').set('looplevelinput', {'all'});
model.result('pg7').feature('lngr1').set('resolution', 'normal');

' Starting the data export' 
model.result.export('data1').set('data', 'cln1');
model.result.export('data1').set('expr', {'V'});
model.result.export('data1').set('unit', {'V'});
model.result.export('data1').set('descr', {'Electric potential'});
model.result.export('data1').set('filename', strcat('/scratch/TrapSimulation_Karan/Axial2QubitTrap_xline_',elecname,'x0-',num2str(x0),'.csv'));
model.result.export('data1').set('location', 'regulargrid');
model.result.export('data1').set('gridstruct', 'spreadsheet');
model.result.export('data1').set('regulargridx1', 40000);
model.result.export('data2').set('data', 'cln2');
model.result.export('data2').set('expr', {'V'});
model.result.export('data2').set('unit', {'V'});
model.result.export('data2').set('descr', {'Electric potential'});
model.result.export('data2').set('filename', strcat('/scratch/TrapSimulation_Karan/Axial2QubitTrap_yline_',elecname,'x0-',num2str(x0),'.csv'));
model.result.export('data2').set('location', 'regulargrid');
model.result.export('data2').set('gridstruct', 'spreadsheet');
model.result.export('data2').set('regulargridx1', 20000);
model.result.export('data3').set('data', 'cln3');
model.result.export('data3').set('expr', {'V'});
model.result.export('data3').set('unit', {'V'});
model.result.export('data3').set('descr', {'Electric potential'});
model.result.export('data3').set('filename', strcat('/scratch/TrapSimulation_Karan/Axial2QubitTrap_zline_',elecname,'x0-',num2str(x0),'.csv'));
model.result.export('data3').set('location', 'regulargrid');
model.result.export('data3').set('gridstruct', 'spreadsheet');
model.result.export('data3').set('regulargridx1', 4000);

model.result.export('data1').set('alwaysask', false);
model.result.export('data1').run;
model.result.export('data1').set('alwaysask', false);
model.result.export('data2').set('alwaysask', false);
model.result.export('data2').run;
model.result.export('data2').set('alwaysask', false);
model.result.export('data3').set('alwaysask', false);
model.result.export('data3').run;
model.result.export('data3').set('alwaysask', false);

% added for the xprime and y prime directions
model.result.export('data7').set('data', 'cln4');
model.result.export('data7').set('expr', {'V'});
model.result.export('data7').set('unit', {'V'});
model.result.export('data7').set('descr', {'Electric potential'});
model.result.export('data7').set('filename', strcat('/scratch/TrapSimulation_Karan/Axial2QubitTrap_y+zline_',elecname,'x0-',num2str(x0),'.csv'));
model.result.export('data7').set('location', 'regulargrid');
model.result.export('data7').set('gridstruct', 'spreadsheet');
model.result.export('data7').set('regulargridx1', 40000);

model.result.export('data8').set('data', 'cln5');
model.result.export('data8').set('expr', {'V'});
model.result.export('data8').set('unit', {'V'});
model.result.export('data8').set('descr', {'Electric potential'});
model.result.export('data8').set('filename', strcat('/scratch/TrapSimulation_Karan/Axial2QubitTrap_y-zline_',elecname,'x0-',num2str(x0),'.csv'));
model.result.export('data8').set('location', 'regulargrid');
model.result.export('data8').set('gridstruct', 'spreadsheet');
model.result.export('data8').set('regulargridx1', 40000);

model.result.export('data7').set('alwaysask', false);
model.result.export('data7').run;
model.result.export('data7').set('alwaysask', false);
model.result.export('data8').set('alwaysask', false);
model.result.export('data8').run;
model.result.export('data8').set('alwaysask', false);

% added for the Ex, Ey, and Ez field exports (for off-diagonal second
% derivatives)
model.result.export('data9').set('data', 'cln1');
model.result.export('data9').set('expr', {'ec.Ey'});
model.result.export('data9').set('unit', {'V/m'});
model.result.export('data9').set('descr', {'Ey'});
model.result.export('data9').set('filename', strcat('/scratch/TrapSimulation_Karan/Axial2QubitTrap_xline_Ey_',elecname,'x0-',num2str(x0),'.csv'));
model.result.export('data9').set('location', 'regulargrid');
model.result.export('data9').set('gridstruct', 'spreadsheet');
model.result.export('data9').set('regulargridx1', 40000);

model.result.export('data10').set('data', 'cln2');
model.result.export('data10').set('expr', {'ec.Ez'});
model.result.export('data10').set('unit', {'V/m'});
model.result.export('data10').set('descr', {'Ez'});
model.result.export('data10').set('filename', strcat('/scratch/TrapSimulation_Karan/Axial2QubitTrap_yline_Ez_',elecname,'x0-',num2str(x0),'.csv'));
model.result.export('data10').set('location', 'regulargrid');
model.result.export('data10').set('gridstruct', 'spreadsheet');
model.result.export('data10').set('regulargridx1', 40000);

model.result.export('data11').set('data', 'cln3');
model.result.export('data11').set('expr', {'ec.Ex'});
model.result.export('data11').set('unit', {'V/m'});
model.result.export('data11').set('descr', {'Ex'});
model.result.export('data11').set('filename', strcat('/scratch/TrapSimulation_Karan/Axial2QubitTrap_zline_Ex_',elecname,'x0-',num2str(x0),'.csv'));
model.result.export('data11').set('location', 'regulargrid');
model.result.export('data11').set('gridstruct', 'spreadsheet');
model.result.export('data11').set('regulargridx1', 40000);


model.result.export('data9').set('alwaysask', false);
model.result.export('data9').run;
model.result.export('data9').set('alwaysask', false);
model.result.export('data10').set('alwaysask', false);
model.result.export('data10').run;
model.result.export('data10').set('alwaysask', false);
model.result.export('data11').set('alwaysask', false);
model.result.export('data11').run;
model.result.export('data11').set('alwaysask', false);

model.modelPath('/scratch/TrapSimulation_Karan');

model.label('2018_12_05_Axial2qubitTrapSim_DC.mph');

if(sflag) % a flag to save the mph file if desired
    model.save(strcat('/scratch/TrapSimulation_Karan/Axial2qubitTrapSim_',elecname,'.mph'))
end
out = model;
