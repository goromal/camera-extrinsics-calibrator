function [] = cam_plot(plot_fig, x_pixels, y_pixels, linecolors, img_w, img_h)

figure(plot_fig);
subplot(1,2,2)
cla;
title('Camera Image Plane')
grid on; hold on

for i=1:length(x_pixels)
    plot(x_pixels(i), img_h - y_pixels(i), string(linecolors(i)), 'LineWidth', 2)
end

xlim([0 img_w])
ylim([0 img_h])
grid on; hold off
pbaspect([img_w img_h 1])

end