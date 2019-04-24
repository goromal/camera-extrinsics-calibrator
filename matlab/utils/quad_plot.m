function [] = quad_plot(plot, time, north, east, down, phi, theta, psi, landmarks, linecolors)
    % visual information
    num_rotors = 4; % reasonable for a quadrotor
    d = 1.0; % Hub displacement from COG
    r = 0.2; % Rotor radius
    s = 10;
    
    global a1s b1s
    if numel(a1s) == 0
        a1s = zeros(1, num_rotors);
        b1s = zeros(1, num_rotors);
    end
    
    for i = 1:num_rotors
        the = (i-1)/num_rotors*2*pi;
        % Rotor hub displacements (1x3)
        % first rotor is on the x-axis, clockwise order looking down from above
        D(:,i) = [d*cos(the); d*sin(the); 0];
    end
    
    % DRAW
    figure(plot);
    subplot(1,2,1)
    cla;
    title(strcat('simulation time: ', string(time)))
    axis([-s/2 s/2 -s/2 s/2 0 s]);
    pbaspect([1 1 1])
    view(80, 20)
    hold on;
    
    % This is the transform from the body frame to the vehicle frame
    phi_ = psi;
    theta_= theta;
    psi_ = phi;
    R = [cos(theta_)*cos(phi_) sin(psi_)*sin(theta_)*cos(phi_)-cos(psi_)*sin(phi_) cos(psi_)*sin(theta_)*cos(phi_)+sin(psi_)*sin(phi_);
         cos(theta_)*sin(phi_) sin(psi_)*sin(theta_)*sin(phi_)+cos(psi_)*cos(phi_) cos(psi_)*sin(theta_)*sin(phi_)-sin(psi_)*cos(phi_);
         -sin(theta_)         sin(psi_)*cos(theta_)                            cos(psi_)*cos(theta_)];      
        
    % CALCULATE FLYER TIP POSITONS USING COORDINATE FRAME ROTATION
    F = [1 0 0;0 -1 0;0 0 -1];
    z = [-north; -east; -down];
        
    % Draw flyer rotors
    t = 0:pi/8:2*pi;
    for j = 1:length(t)
        circle(:,j) = [r*sin(t(j));r*cos(t(j));0]; %#ok<*AGROW>
    end
        
    for i = 1:num_rotors
        hub(:,i) = F*(-z + R*D(:,i)); %points in the inertial frame
            
        q = 1; % Flapping angle scaling for output display - makes it easier to see what flapping is occurring
        Rr = [cos(q*a1s(i))  sin(q*b1s(i))*sin(q*a1s(i)) cos(q*b1s(i))*sin(q*a1s(i));
              0              cos(q*b1s(i))               -sin(q*b1s(i));
              -sin(q*a1s(i)) sin(q*b1s(i))*cos(q*a1s(i)) cos(q*b1s(i))*cos(q*a1s(i))];
            
        tippath(:,:,i) = F*R*Rr*circle;
        plot3(hub(1,i) + tippath(1,:,i), hub(2,i) + tippath(2,:,i), hub(3,i) + tippath(3,:,i),'k-')
    end
        
    % Draw flyer
    hub0 = z; % center of vehicle
    for i = 1:num_rotors
        % line from hub to center
        plot3([hub(1,i) -hub0(1)],[hub(2,i) hub0(2)],[hub(3,i) hub0(3)],'-k')
            
        % plot a circle at the hub itself
        plot3(hub(1,i), hub(2,i), hub(3,i), 'ko')
    end
        
    % plot the vehicle's centroid on the ground plane
    plot3([-z(1) 0], [z(2) 0], [0 0], '--k')
    plot3(-z(1), z(2), 0, 'xk') 
    
    % plot the ground boundaries and the big cross
    plot3([-s -s],[s -s],[0 0],'--k')
    plot3([-s s],[s s],[0 0],'--k')
    plot3([s -s],[-s -s],[0 0],'--k')
    plot3([s s],[s -s],[0 0],'--k')
    grid on
    
    R_veh_to_veh1 = [cos(psi), sin(psi), 0; -sin(psi), cos(psi), 0; 0, 0, 1];
    R_veh1_to_veh2 = [cos(theta), 0, -sin(theta); 0, 1, 0; sin(theta), 0, cos(theta)];
    R_veh2_to_body = [1, 0, 0; 0, cos(phi), sin(phi); 0, -sin(phi), cos(phi)];
    R_veh_to_body = R_veh2_to_body * R_veh1_to_veh2 * R_veh_to_veh1;
        
    R_veh_to_plot = [1, 0, 0; 0, -1, 0; 0, 0, -1];
        
    body_forward = [1; 0; 0];
    body_right = [0; 1; 0];
    body_down = [0; 0; 1];
    inertial_forward = R_veh_to_plot * R_veh_to_body' * body_forward;
    inertial_right = R_veh_to_plot * R_veh_to_body' * body_right;
    inertial_down = R_veh_to_plot * R_veh_to_body' * body_down;
        
    veh_position = [north; east; down];
    plot_position = R_veh_to_plot * veh_position;
        
    % Plot body forward, right, down axes
    plot3([plot_position(1) plot_position(1) + inertial_forward(1)],...
          [plot_position(2) plot_position(2) + inertial_forward(2)], ...
          [plot_position(3) plot_position(3) + inertial_forward(3)], 'r')
    plot3([plot_position(1) plot_position(1) + inertial_right(1)], ...
          [plot_position(2) plot_position(2) + inertial_right(2)], ...
          [plot_position(3) plot_position(3) + inertial_right(3)], 'g')
    plot3([plot_position(1) plot_position(1) + inertial_down(1)], ...
          [plot_position(2) plot_position(2) + inertial_down(2)], ...
          [plot_position(3) plot_position(3) + inertial_down(3)], 'b')
      
    plot_landmarks = R_veh_to_plot * landmarks';
    plot_landmarks = plot_landmarks';
    
    for i=1:size(plot_landmarks,1)
       plot3(plot_landmarks(i,1),plot_landmarks(i,2),plot_landmarks(i,3),...
           string(linecolors(i)));
    end
    
    % label the axes
    xlabel('N');
    ylabel('-E');
    zlabel('-D (height above ground)');
        