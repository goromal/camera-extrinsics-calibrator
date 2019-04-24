% This file extracts data from the .mat file from Planck and writes the 
% data to logs to be interpreted by the calibrator.

%% Setup

if ~exist('logs', 'dir')
    mkdir('logs');
end
data_filename = 'planck_ctrl_2019-02-22-21-53-35.mat'; % given Planck data
addpath ../matlab/utils/ % more helper functions

visualize = false; % visualize an animation of the log data?

%% Data Extraction

% load raw data from Planck
load(data_filename);

% state data from sensor fusion
kfp_t       = extract_label(KFP, KFP_label, 'time(us)') / 1e6;
kfp_n       = extract_label(KFP, KFP_label, 'est_NED_x (m)');
kfp_e       = extract_label(KFP, KFP_label, 'est_NED_y (m)');
kfp_d       = extract_label(KFP, KFP_label, 'est_NED_z (m)');

% convert data from vehicle-centered frame to inertial frame
kfp_n       = -1.0 * kfp_n;
kfp_e       = -1.0 * kfp_e;
kfp_d       = -1.0 * kfp_d;

% extract raw attitude data
att_t       = extract_label(DR_ATT, DR_ATT_label, 'time(us)') / 1e6;
att_phi     = extract_label(DR_ATT, DR_ATT_label, 'roll (rad)');
att_theta   = extract_label(DR_ATT, DR_ATT_label, 'pitch (rad)');
att_psi     = extract_label(DR_ATT, DR_ATT_label, 'yaw (rad)');

% make attitude data continuous
att_phi     = make_continuous_angles(att_phi);
att_theta   = make_continuous_angles(att_theta);
att_psi     = make_continuous_angles(att_psi);

% extract feature measurements
ft_t        = extract_label(AT, AT_label, 'time(us)') / 1e6;
ft_1x       = extract_label(AT, AT_label, 'x1 (pixels)');
ft_1y       = extract_label(AT, AT_label, 'y1 (pixels)');
ft_2x       = extract_label(AT, AT_label, 'x2 (pixels)');
ft_2y       = extract_label(AT, AT_label, 'y2 (pixels)');
ft_3x       = extract_label(AT, AT_label, 'x3 (pixels)');
ft_3y       = extract_label(AT, AT_label, 'y3 (pixels)');
ft_4x       = extract_label(AT, AT_label, 'x4 (pixels)');
ft_4y       = extract_label(AT, AT_label, 'y4 (pixels)');
ft_0x       = (ft_1x + ft_2x + ft_3x + ft_4x) / 4;
ft_0y       = (ft_1y + ft_2y + ft_3y + ft_4y) / 4;

% extract landmark data
tag_size    = extract_label(AT, AT_label, 'tag_size (m)');
tag_size    = tag_size(1); % tag side length in meters
th_TFNED    = pi; % transform between tag frame and inertial frame
R_TF_NED    = [cos(th_TFNED) sin(th_TFNED) 0;...
              -sin(th_TFNED) cos(th_TFNED) 0;...
                           0             0 1];
tag_c_TF    = [tag_size tag_size -tag_size -tag_size;...
              -tag_size tag_size  tag_size -tag_size;...
                      0        0         0         0]/2;
tag_c_NED   = R_TF_NED * tag_c_TF;
tag_landmarks = [0 0 0; tag_c_NED']';

% extract camera data for animation (and for the user's information)
img_w       = extract_label(INI, INI_label, 'camera_width');
img_h       = extract_label(INI, INI_label, 'camera_height');
fx_true     = extract_label(INI, INI_label, 'camera_f_len'); % focal length

%% Data Regularization

% make all data common time
t0_common      = max([kfp_t(1) att_t(1) ft_t(1)]);
tf_common      = min([kfp_t(end) att_t(end) ft_t(end)]);
samples_common = min([length(kfp_t), length(att_t), length(ft_t)]);
t_meas         = linspace(t0_common, tf_common, samples_common);

% interpolate state data
N_meas         = interp1(kfp_t, kfp_n, t_meas);
E_meas         = interp1(kfp_t, kfp_e, t_meas);
D_meas         = interp1(kfp_t, kfp_d, t_meas);
r_meas         = interp1(att_t, att_phi, t_meas);
p_meas         = interp1(att_t, att_theta, t_meas);
y_meas         = interp1(att_t, att_psi, t_meas);

% interpolate feature data
ft_0id_meas     = 0 * ones(size(t_meas));
ft_0x_meas      = interp1(ft_t, ft_0x, t_meas);
ft_0y_meas      = interp1(ft_t, ft_0y, t_meas);
ft_1id_meas     = 1 * ones(size(t_meas));
ft_1x_meas      = interp1(ft_t, ft_1x, t_meas);
ft_1y_meas      = interp1(ft_t, ft_1y, t_meas);
ft_2id_meas     = 2 * ones(size(t_meas));
ft_2x_meas      = interp1(ft_t, ft_2x, t_meas);
ft_2y_meas      = interp1(ft_t, ft_2y, t_meas);
ft_3id_meas     = 3 * ones(size(t_meas));
ft_3x_meas      = interp1(ft_t, ft_3x, t_meas);
ft_3y_meas      = interp1(ft_t, ft_3y, t_meas);
ft_4id_meas     = 4 * ones(size(t_meas));
ft_4x_meas      = interp1(ft_t, ft_4x, t_meas);
ft_4y_meas      = interp1(ft_t, ft_4y, t_meas);

% normalize time
t_meas = t_meas - t_meas(1);

%% Data Packaging

num_meas = length(t_meas);

% Make attitude data conversions
[qw_meas, qx_meas, qy_meas, qz_meas] = euler_to_quat_vecs(r_meas, p_meas, y_meas);

% Combine pose data together
state = zeros(8, num_meas);
for i = 1:num_meas
    state(1, i) = t_meas(i);
    state(2, i) = N_meas(i);
    state(3, i) = E_meas(i);
    state(4, i) = D_meas(i);
    state(5, i) = qw_meas(i);
    state(6, i) = qx_meas(i);
    state(7, i) = qy_meas(i);
    state(8, i) = qz_meas(i);
end

% Combine feature data together
features = zeros(4, 5 * num_meas);
for i = 1:num_meas
    idx = (i - 1) * 5 + 1;
    features(1, idx) = t_meas(i);
    features(2, idx) = ft_0id_meas(i);
    features(3, idx) = ft_0x_meas(i);
    features(4, idx) = ft_0y_meas(i);
    features(1, idx+1) = t_meas(i);
    features(2, idx+1) = ft_1id_meas(i);
    features(3, idx+1) = ft_1x_meas(i);
    features(4, idx+1) = ft_1y_meas(i);
    features(1, idx+2) = t_meas(i);
    features(2, idx+2) = ft_2id_meas(i);
    features(3, idx+2) = ft_2x_meas(i);
    features(4, idx+2) = ft_2y_meas(i);
    features(1, idx+3) = t_meas(i);
    features(2, idx+3) = ft_3id_meas(i);
    features(3, idx+3) = ft_3x_meas(i);
    features(4, idx+3) = ft_3y_meas(i);
    features(1, idx+4) = t_meas(i);
    features(2, idx+4) = ft_4id_meas(i);
    features(3, idx+4) = ft_4x_meas(i);
    features(4, idx+4) = ft_4y_meas(i);
end

% True camera offsets; NOT SURE IF THESE ARE RIGHT, better check with
% Planck!
offsets = [0; 0; -0.15; euler_to_quat([0, 0, pi/2])'];

%% Log Writing

write_log(offsets,       'logs/offsets_Planck.log');
write_log(state,         'logs/states_Planck.log');
write_log(features,      'logs/features_Planck.log');
write_log(tag_landmarks, 'logs/landmarks_Planck.log');

%% Simulation and Visualization (if toggled)

if(visualize)
    % cycle through points
    H = figure('Position', [10 10 1200 900],'units','normalized','outerposition',[0 0 1 1]); %#ok<*UNRCH>
    
    dt = 0.005; % pause this amount of time after drawing each one
    linecolors = {'kx','rx','gx','bx','cx'}; % landmark/pixel measurement colors

    for i = 1:length(t_meas)
  
        quad_plot(H, t_meas(i), N_meas(i), E_meas(i), D_meas(i), r_meas(i), ...
                  p_meas(i), y_meas(i), tag_landmarks', linecolors);
        
        % process feature measurements and visualize
        x_pixel = [ft_0x_meas(i) ft_1x_meas(i) ft_2x_meas(i)...
                   ft_3x_meas(i) ft_4x_meas(i)];
        y_pixel = [ft_0y_meas(i) ft_1y_meas(i) ft_2y_meas(i)...
                   ft_3y_meas(i) ft_4y_meas(i)];
        if ~isnan(x_pixel)
            cam_plot(H, x_pixel, y_pixel, linecolors, img_w, img_h);
        end
        pause(dt) 
    end  
end

%% Supporting Functions

function data = extract_label(data_vector, str_array, str_label)

data = data_vector(:, find(strcmp(str_array, str_label))); %#ok<*FNDSB>

end

function angles = make_continuous_angles(angles)

epsilon = 3 * pi / 2;
n = length(angles);
for i = 2:n
    
    if ((angles(i) - angles(i-1)) < -epsilon)
        angles(i) = angles(i) + 2 * pi;
    elseif ((angles(i) - angles(i-1)) > epsilon)
        angles(i) = angles(i) - 2 * pi;
    end
    
end

end