%% current_loop_design.m  - optional MATLAB cross-check of the Python charger loop (Phase 2)
% Reproduces the analytic small-signal model of src/cdrt/charger.py:
% synchronous buck, average-current-mode digital PI, 1.5*Ts delay, inductor sensing.
% Requires Control System Toolbox. The project does NOT depend on this file.
% Battery pack is approximated by a 1 Ohm resistor here (the Python model uses the PyBaMM
% impedance x 32); results at the crossover change by < 1 % because sL dominates there.
clear; s = tf('s');
Vdc = 450; fsw = 100e3; Ts = 1/fsw; L = 470e-6; RL = 30e-3;
Co = 10e-6; ESR = 10e-3; Rpack = 1.0;
fc = 5e3; Kp = 2*pi*fc*L/Vdc; Ki = Kp*2*pi*fc/10;
Zc = 1/(s*Co) + ESR;  Zo = Zc*Rpack/(Zc + Rpack);
Gid = Vdc/(s*L + RL + Zo);                 % duty -> inductor current
Hd  = exp(-1.5*Ts*s);                       % computation + PWM delay
Hs  = 1/(1 + s/(2*pi*50e3));                % current-sensor bandwidth
C   = Kp + Ki/s;
T   = C*Hd*Gid*Hs;                          % loop gain
[Gm, Pm, Wcg, Wcp] = margin(T);
fprintf('Kp = %.4f, Ki = %.2f\n', Kp, Ki);
fprintf('crossover %.0f Hz, PM %.1f deg, GM %.1f dB\n', Wcp/2/pi, Pm, 20*log10(Gm));
Tcl  = feedback(C*Hd*Gid, Hs);              % reference -> inductor current
kbat = Zc/(Zc + Rpack);                     % inductor -> battery current divider
figure; bode(T, Tcl*kbat, {2*pi*1, 2*pi*3e4}); grid on;
legend('loop gain', 'i_{bat}/i_{ref}');
% Python reference values (results/phase2_summary.json): f_cross 5001 Hz, PM 52.4 deg, GM 8.9 dB
