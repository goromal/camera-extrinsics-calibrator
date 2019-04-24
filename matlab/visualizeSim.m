% Reads the logs from the simulator (unit test 1) and visualizes the UAV 
% flight path and feature measurements

addpath utils/ % helper functions
log_directory = '../logs/';

%% Import log data

% Camera image size (should match ../params/sim_params.yaml)
img_w = 640;
img_h = 480;

% Load simulation truth data
truth_log = fopen(strcat(log_directory, 'Simulated.x_truth.log'), 'r');
truth = fread(truth_log, 'double');
truth = reshape(truth, 8, []);

t  = truth(1, :);
N  = truth(2, :);
E  = truth(3, :);
D  = truth(4, :);
qw = truth(5, :);
qx = truth(6, :);
qy = truth(7, :);
qz = truth(8, :);
[r, p, y] = quat_to_euler_vecs(qw, qx, qy, qz);

% Load landmarks
landmarks_log = fopen(strcat(log_directory, 'Simulated.Features.Landmarks.log'), 'r');
landmarks = fread(landmarks_log, 'double');
landmarks = reshape(landmarks, 3, []);

% load simulation features
sim_feat_log = fopen(strcat(log_directory, 'Simulated.Features.Truth.log'), 'r');
sim_feat = fread(sim_feat_log, 'double');
sim_feat = reshape(sim_feat, 4, []);

% get pixel measurements for each feature id
t_s_ID0 = [];
u_s_ID0 = [];
v_s_ID0 = [];
t_s_ID1 = [];
u_s_ID1 = [];
v_s_ID1 = [];
t_s_ID2 = [];
u_s_ID2 = [];
v_s_ID2 = [];
t_s_ID3 = [];
u_s_ID3 = [];
v_s_ID3 = [];
t_s_ID4 = [];
u_s_ID4 = [];
v_s_ID4 = [];

for i = 1:size(sim_feat, 2)
    t_feat = sim_feat(1, i);
    id = sim_feat(2, i);
    if id == 0
        t_s_ID0 = [t_s_ID0 t_feat]; %#ok<*AGROW>
        u_s_ID0 = [u_s_ID0 sim_feat(3, i)];
        v_s_ID0 = [v_s_ID0 sim_feat(4, i)];
    elseif id == 1
        t_s_ID1 = [t_s_ID1 t_feat];
        u_s_ID1 = [u_s_ID1 sim_feat(3, i)];
        v_s_ID1 = [v_s_ID1 sim_feat(4, i)];
    elseif id == 2
        t_s_ID2 = [t_s_ID2 t_feat];
        u_s_ID2 = [u_s_ID2 sim_feat(3, i)];
        v_s_ID2 = [v_s_ID2 sim_feat(4, i)];
    elseif id == 3
        t_s_ID3 = [t_s_ID3 t_feat];
        u_s_ID3 = [u_s_ID3 sim_feat(3, i)];
        v_s_ID3 = [v_s_ID3 sim_feat(4, i)];
    elseif id == 4
        t_s_ID4 = [t_s_ID4 t_feat];
        u_s_ID4 = [u_s_ID4 sim_feat(3, i)];
        v_s_ID4 = [v_s_ID4 sim_feat(4, i)];
    end
end

ref_data = u_s_ID0;
if isempty(u_s_ID1)
    t_s_ID1 = zeros(size(ref_data));
    u_s_ID1 = zeros(size(ref_data));
    v_s_ID1 = zeros(size(ref_data));
end
if isempty(u_s_ID2)
    t_s_ID2 = zeros(size(ref_data));
    u_s_ID2 = zeros(size(ref_data));
    v_s_ID2 = zeros(size(ref_data));
end
if isempty(u_s_ID3)
    t_s_ID3 = zeros(size(ref_data));
    u_s_ID3 = zeros(size(ref_data));
    v_s_ID3 = zeros(size(ref_data));
end
if isempty(u_s_ID4)
    t_s_ID4 = zeros(size(ref_data));
    u_s_ID4 = zeros(size(ref_data));
    v_s_ID4 = zeros(size(ref_data));
end

%% Interpolate data

% Find the feature id with the least measurements
feature_lengths = [length(t_s_ID0) length(t_s_ID1) length(t_s_ID2) ...
                   length(t_s_ID3) length(t_s_ID4)];
[~, min_idx] = min(feature_lengths);
             
% Align all measurements to this least common denominator
if min_idx == 1
    t_meas = t_s_ID0;
elseif min_idx == 2
    t_meas = t_s_ID1;
elseif min_idx == 3
    t_meas = t_s_ID2;
elseif min_idx == 4
    t_meas = t_s_ID3;
elseif min_idx == 5
    t_meas = t_s_ID4;
end

N_meas = interp1(t, N, t_meas);
E_meas = interp1(t, E, t_meas);
D_meas = interp1(t, D, t_meas);
r_meas = interp1(t, r, t_meas);
p_meas = interp1(t, p, t_meas);
y_meas = interp1(t, y, t_meas);

u_s_ID0_meas = interp1(t_s_ID0, u_s_ID0, t_meas);
v_s_ID0_meas = interp1(t_s_ID0, v_s_ID0, t_meas);
u_s_ID1_meas = interp1(t_s_ID1, u_s_ID1, t_meas);
v_s_ID1_meas = interp1(t_s_ID1, v_s_ID1, t_meas);
u_s_ID2_meas = interp1(t_s_ID2, u_s_ID2, t_meas);
v_s_ID2_meas = interp1(t_s_ID2, v_s_ID2, t_meas);
u_s_ID3_meas = interp1(t_s_ID3, u_s_ID3, t_meas);
v_s_ID3_meas = interp1(t_s_ID3, v_s_ID3, t_meas);
u_s_ID4_meas = interp1(t_s_ID4, u_s_ID4, t_meas);
v_s_ID4_meas = interp1(t_s_ID4, v_s_ID4, t_meas);

%% Animate
% cycle through points
H = figure('Position', [10 10 1200 900],'units','normalized','outerposition',[0 0 1 1]); %#ok<*UNRCH>
    
dt = 0.001; % pause this amount of time after drawing each one
linecolors = {'kx','rx','gx','bx','cx'}; % landmark/pixel measurement colors

for i = 1:length(t_meas)
    quad_plot(H, t_meas(i), N_meas(i), E_meas(i), D_meas(i), r_meas(i), ...
              p_meas(i), y_meas(i), landmarks', linecolors);        
    % process feature measurements and visualize
    x_pixel = [u_s_ID0_meas(i) u_s_ID1_meas(i) u_s_ID2_meas(i) ...
               u_s_ID3_meas(i) u_s_ID4_meas(i)];
    y_pixel = [v_s_ID0_meas(i) v_s_ID1_meas(i) v_s_ID2_meas(i) ...
               v_s_ID3_meas(i) v_s_ID4_meas(i)];
    if ~isnan(x_pixel)
        cam_plot(H, x_pixel, y_pixel, linecolors, img_w, img_h);
    end
    pause(dt) 
end