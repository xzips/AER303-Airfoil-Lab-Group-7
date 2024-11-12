f_s=1000;
time=10;

[PT_data] = Meas_T_PS(f_s,time);

fileName=input("enter the file name: ","s");

save(strcat(fileName,".mat"))