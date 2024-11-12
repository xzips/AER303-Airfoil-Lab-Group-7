function [PT_data] = Meas_T_PS(f_s,time)
    

        % Configurate the data acquisition
%         f_s = 4000;
%         time = 30;
        s1 = daq.createSession('ni');
        s1.Rate=f_s;
        s1.DurationInSeconds=time;
        chans_PT = s1.addAnalogInputChannel('Dev4',1,'Voltage');
        set(s1.Channels,'TerminalConfig','Differential');
       
    
        % Start sampling
        PT_data=startForeground(s1);
        plot(PT_data);
        mean_V = mean(PT_data);
        
        % Clear DAQ from system
        delete(s1);
        
     
end

