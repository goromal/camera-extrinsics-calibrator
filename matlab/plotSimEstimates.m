% Compares the UAV state estimate of the EKF with truth from the simulator 
% (unit test 1) logs

addpath utils/ % helper functions
log_directory = '../logs/';

%% Import log data

% Simulation truth data
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

% Simulation estimate data
est_log = fopen(strcat(log_directory, 'Simulated.x_est.log'), 'r');
est = fread(est_log, 'double');
est = reshape(est, 8, []);

t_est  = est(1, :);
N_est  = est(2, :);
E_est  = est(3, :);
D_est  = est(4, :);
qw_est = est(5, :);
qx_est = est(6, :);
qy_est = est(7, :);
qz_est = est(8, :);
[r_est, p_est, y_est] = quat_to_euler_vecs(qw_est, qx_est, qy_est, qz_est);

%% Plot comparison

subplot(2, 3, 1)
plot(t, N, 'k-', t_est, N_est, 'r--')
ylabel('North (m)')
xlabel('t (s)')
legend('truth','estimate')
subplot(2, 3, 2)
plot(t, E, 'k-', t_est, E_est, 'r--')
ylabel('East (m)')
xlabel('t (s)')
subplot(2, 3, 3)
plot(t, D, 'k-', t_est, D_est, 'r--')
ylabel('Down (m)')
xlabel('t (s)')
subplot(2, 3, 4)
plot(t, r, 'k-', t_est, r_est, 'r--')
ylabel('roll (rad)')
xlabel('t (s)')
subplot(2, 3, 5)
plot(t, p, 'k-', t_est, p_est, 'r--')
ylabel('pitch (rad)')
xlabel('t (s)')
subplot(2, 3, 6)
plot(t, y, 'k-', t_est, y_est, 'r--')
ylabel('yaw (rad)')
xlabel('t (s)')