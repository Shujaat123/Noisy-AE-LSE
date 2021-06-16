clc
clear all
close all

load('Noisy_AE_PSNR_Results.mat')

figure(1)
hold on
for i=1:2:6
    plot(PSNR_Results(:,i),PSNR_Results(:,i+1),'o--')
end
plot([10 35],[10 35])
xlabel('PSNR_{in} (dB)')
ylabel('PSNR_{out} (dB)')
title('Autoencoder(AE) [Noise Characteristics Curve]')
legend('Uniform','Gaussian','Speckle','in==out','location','best')
grid minor
xlim([min(min(PSNR_Results(:,1:2:end)))-1 max(max(PSNR_Results(:,1:2:end)))])
ylim([min(min(PSNR_Results(:,2:2:end)))-1 max(max(PSNR_Results(:,2:2:end)))+2])


