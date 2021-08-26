
ElecNams = {'DC1','DC2','DC3','DC4','DC5','DC6','DC7','DC8','DC9','DC10'}; 
ElecDoms = {[15 30 66 96 112],[104 105],[100 101],[87 88],...
    [83 84],[79 81],[75 77],[48 50],[42 46],[40 44]}; 
    % cell array of domain numbers for the electrodes indexed to the cell
    % array

sflag = 0; 
x0 = 0; 
for k=1:numel(ElecNams)
%   if(k==1)
    RunOneElec(ElecNams{k}, ElecDoms{k}, x0, sflag); 
%   end
endn