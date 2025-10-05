%% Group14; Adaptive Noise Cancellation
mode = 1; % 1 = Full Suppression, 0 = Partial Suppression

%% Load Data
fs = 44100;
noisy_speech = load('noisy_speech.txt');     % s(n) + v(n)
external_noise = load('external_noise.txt'); % w(n)
clean_speech = load('clean_speech.txt');     % s(n)

L = length(noisy_speech);
residual = zeros(L,1);

if mode == 0
    %% ---------------- Partial Suppression ----------------
    fprintf("Running Partial Suppression Mode...\n");

    f_notch = 1000;
    r = 0.99;
    omega = 2 * pi * f_notch / fs;
    b = [1, -2*cos(omega), 1];
    a = [1, -2*r*cos(omega), r^2];

    % Notch filter states
    x_prev = [0, 0]; % w[n-1], w[n-2]
    y_prev = [0, 0]; % y[n-1], y[n-2]

    % RLS Param
    N = 5;                       
    lambda = 0.998;           
    delta = 0.01;
    P = (1/delta) * eye(N);      
    W = zeros(N,1);              
    w_buffer = zeros(N,1);

    for n = 1:L
        x_n = external_noise(n);
        y_n = b(1)*x_n + b(2)*x_prev(1) + b(3)*x_prev(2) - a(2)*y_prev(1) - a(3)*y_prev(2);
        x_prev = [x_n, x_prev(1)];
        y_prev = [y_n, y_prev(1)];

        w_buffer = [y_n; w_buffer(1:end-1)];
        if n >= N
            u = w_buffer;
            pi_vec = P * u;
            k = pi_vec / (lambda + u' * pi_vec);
            y_hat = W' * u;
            e = noisy_speech(n) - y_hat;
            W = W + k * e;
            P = (P - k * u' * P) / lambda;
            residual(n) = e;
        else
            residual(n) = noisy_speech(n);
        end
    end

    filename = 'cleaned_speech_partial.txt';

else
    %% ---------------- Full Suppression ----------------
    fprintf("Running Full Suppression Mode...\n");

    % RLS Param
    M_RLS = 5;
    lambda_RLS = 0.998;
    delta_RLS = 0.01;

    w_filter_RLS = zeros(M_RLS, 1);
    P_RLS = (1/delta_RLS) * eye(M_RLS);
    x_buffer_RLS = zeros(M_RLS, 1);

    %Conventional RLS Algorithm (Reference:  Paulo S. R. Diniz, ”Adaptive
    %Filtering: Algorithms and Practical Implementation", 5/e,
    %Springer,2020, Chapter 5)

    for n = M_RLS:L
        x_buffer_RLS = [external_noise(n); x_buffer_RLS(1:end-1)];
        y_hat_RLS = w_filter_RLS' * x_buffer_RLS;
        e = noisy_speech(n) - y_hat_RLS;
        k_num = P_RLS * x_buffer_RLS;
        k_den = lambda_RLS + x_buffer_RLS' * k_num;
        k_RLS = k_num / k_den;
        w_filter_RLS = w_filter_RLS + k_RLS * e;
        P_RLS = (P_RLS - k_RLS * x_buffer_RLS' * P_RLS) / lambda_RLS;
        residual(n) = e;
    end

    filename = 'cleaned_speech_full.txt';

end

%% Save and Play Audio
residual = residual(:);
save(filename, 'residual', '-ascii');
sound(residual, fs);

%% Spectrograms
figure;
subplot(2,1,1);
spectrogram(noisy_speech, 512, 384, 1024, fs, 'yaxis');
title('Spectrogram of Noisy Speech');
colorbar;

subplot(2,1,2);
spectrogram(residual, 512, 384, 1024, fs, 'yaxis');
title(['Spectrogram After ', strrep(filename, '_', '\_')]);
colorbar;

%% SNR Evaluation
noise_before = noisy_speech - clean_speech;
noise_after = residual - clean_speech;

P_speech = mean(clean_speech.^2);      
P_noise_before = mean(noise_before.^2); 
P_noise_after = mean(noise_after.^2);

snr_before = 10 * log10(P_speech / P_noise_before); 
snr_after = 10 * log10(P_speech / P_noise_after);
snr_gain = snr_after - snr_before;

fprintf('SNR before: %.2f dB\n', snr_before);
fprintf('SNR after:  %.2f dB\n', snr_after);
fprintf('SNR gain:   %.2f dB\n', snr_gain);

%% FFT Comparison
NFFT = 2^nextpow2(L);
f_axis = fs*(0:(NFFT/2))/NFFT;

X = fft(noisy_speech, NFFT);
X_mag = abs(X / L);
X_mag = X_mag(1:NFFT/2+1);

R = fft(residual, NFFT);
R_mag = abs(R / L);
R_mag = R_mag(1:NFFT/2+1);

figure;
plot(f_axis, 20*log10(X_mag + eps), 'b', 'DisplayName', 'Noisy Speech');
hold on;
plot(f_axis, 20*log10(R_mag + eps), 'r', 'DisplayName', 'Residual Output');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('FFT Comparison');
legend;
grid on;
xlim([0 20000]);

%% Time Domain Comparison
t = (0:L-1)/fs;

figure;
subplot(2,1,1);
plot(t, noisy_speech);
title('Noisy Speech (Time Domain)');
xlabel('Time (s)'); ylabel('Amplitude');
grid on; xlim([0 12]);

subplot(2,1,2);
plot(t, residual);
title(['Residual (Time Domain) - ', strrep(filename, '_', '\_')]);
xlabel('Time (s)'); ylabel('Amplitude');
grid on; xlim([0 12]);

%% Notch Filter freq.response plot
% fs = 44100;           
% f_notch = 1000;       
% r = 0.98;             
% omega = 2 * pi * f_notch / fs;
% b = [1, -2*cos(omega), 1];
% a = [1, -2*r*cos(omega), r^2];
% N_fft = 1024;
% [H, f] = freqz(b, a, N_fft, fs);
% 
% % Plot magnitude response
% figure;
% subplot(2,1,1);
% plot(f, 20*log10(abs(H)), 'LineWidth', 1.5);
% grid on;
% xlabel('Frequency (Hz)');
% ylabel('Magnitude (dB)');
% title(sprintf('Magnitude Response of Notch Filter (%.0f Hz)', f_notch));
% xlim([0 fs/2]);
% 
% % Plot phase response
% subplot(2,1,2);
% plot(f, angle(H)*180/pi, 'LineWidth', 1.5);
% grid on;
% xlabel('Frequency (Hz)');
% ylabel('Phase (degrees)');
% title('Phase Response');
% xlim([0 fs/2]);

%% Partial Suppression (Metric)
%fprintf('\n--- Post-Notch Evaluation of Partial Suppression Output ---\n');

residual_post_notch = filter(b, a, residual);

% Evaluate SNR
noise_post = residual_post_notch - clean_speech;
P_noise_post = mean(noise_post.^2);
snr_post = 10 * log10(P_speech / P_noise_post);

% fprintf('SNR after post-notch (on residual): %.2f dB\n', snr_post);
% fprintf('SNR from full suppression:         %.2f dB\n', snr_after);
% fprintf('Difference in SNR:                 %.2f dB\n', abs(snr_after - snr_post));
