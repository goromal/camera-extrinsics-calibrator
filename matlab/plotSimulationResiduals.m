% Loads the logs from the simulator (unit test 1) and plots the pixel
% measurement residuals from the optimizer

addpath utils/ % helper functions
log_directory = '../logs/';

%% Import log data

% Camera image size (should match ../params/sim_params.yaml)
img_w = 640;
img_h = 480;

% Load simulation feature measurements
feat_log = fopen(strcat(log_directory, 'Simulated.Features.Truth.log'), 'r');
feat = fread(feat_log, 'double');
feat = reshape(feat, 4, []);

% Load simulation initial residuals
res0_log = fopen(strcat(log_directory, 'Simulation.PixRes0.log'), 'r');
res0 = fread(res0_log, 'double');
res0 = reshape(res0, 2, []);

% Load simulation final residuals
resf_log = fopen(strcat(log_directory, 'Simulation.PixResf.log'), 'r');
resf = fread(resf_log, 'double');
resf = reshape(resf, 2, []);

%% Process data

% Get pixel measurements and calculate estimates for each feature id
u_s = [];
v_s = [];
u_s_est = [];
v_s_est = [];

for i = 1:size(feat, 2)
    u_s = [u_s feat(3, i)]; %#ok<*AGROW>
    v_s = [v_s feat(4, i)];
    u_s_est = [u_s_est (u_s(end) - res0(1, i))];
    v_s_est = [v_s_est (v_s(end) - res0(2, i))];
end

%% Plot residuals

figure(1)
grid on
plot(u_s, img_h - v_s, 'b.', u_s_est, img_h - v_s_est, 'k.')
xlim([0 img_w])
ylim([0 img_h])
xlabel('x (pixels)')
ylabel('y (pixels)')
pbaspect([img_w img_h 1])
title('Camera Measurements vs Estimates')
legend('measurement','estimate')

figure(2)
grid on
plot(res0(1, :), res0(2, :), 'k.', 0, 0, 'ro')
xlim([min([res0(1, :) 0]) max([res0(1, :) 0])])
ylim([min([res0(2, :) 0]) max([res0(2, :) 0])])
xlabel('x (pixels)')
ylabel('y (pixels)')
pbaspect([range([res0(1, :) 0]) range([res0(2, :) 0]) 1])
title('Measurement Residuals Before Optimization')

figure(3)
grid on
plot(resf(1, :), resf(2, :), 'k.', 0, 0, 'ro')
xlim([min([resf(1, :) 0]) max([resf(1, :) 0])])
ylim([min([resf(2, :) 0]) max([resf(2, :) 0])])
xlabel('x (pixels)')
ylabel('y (pixels)')
pbaspect([range([resf(1, :) 0]) range([resf(2, :) 0]) 1])
title('Measurement Residuals After Optimization')