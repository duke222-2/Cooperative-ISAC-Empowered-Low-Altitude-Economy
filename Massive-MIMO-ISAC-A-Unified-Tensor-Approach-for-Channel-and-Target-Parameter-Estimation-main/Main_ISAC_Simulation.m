% 主程序：空间平滑 CPD 目标参数估计 (Monte Carlo 性能评估)
clc; clear; close all;

%% 1. 系统参数设置 (System Settings)
params = struct();
params.fc = 4.9e9;           
params.c0 = 3e8;             
params.lambda = params.c0 / params.fc;       
params.BW = 20e6;            
params.M = 300;              % 按照第一版代码设为 300
params.N = 7;                
params.delta_f = 30e3;       
params.Ts = 35.677e-6;       
params.P = 16; params.Q = 24; 
params.L = params.P * params.Q;            
params.R = 64;               

% 仿真遍历参数
P_tx_dBm = 45:5:65;
MC_trials = 100;      % 跑图建议 100+
K_list = [2, 4];

% 目标真实参数 (最大支持4个目标)
targets_all.d = [180, 30, 100, 150];
targets_all.v = [10, -20, 5, -10];
targets_all.th = deg2rad([10, 25, 40, 55]);
targets_all.ph = deg2rad([30, 60, 15, 45]);

% 初始化结果存储结构体
RES = struct('aoa', zeros(2, length(P_tx_dBm)), ...
             'range', zeros(2, length(P_tx_dBm)), ...
             'vel', zeros(2, length(P_tx_dBm)), ...
             'pos', zeros(2, length(P_tx_dBm)));

%% 2. 核心 Monte Carlo 仿真循环
for k_idx = 1:length(K_list)
    K = K_list(k_idx);
    
    % 为当前 K 截取有效的目标参数
    t.d = targets_all.d(1:K);
    t.v = targets_all.v(1:K);
    t.th = targets_all.th(1:K);
    t.ph = targets_all.ph(1:K);
    
    for p_idx = 1:length(P_tx_dBm)
        curr_P = P_tx_dBm(p_idx);
        fprintf('Calculating K=%d, Power=%d dBm ', K, curr_P);
        
        err_sq = zeros(1, 4); % 累加器: [range, vel, aoa, pos]
        
        for mc = 1:MC_trials
            % A. 信号生成 (SNR随功率变化)
            snr = curr_P - 45;
            [Y, Frx_c] = gen_sig_tensor(params, t, K, snr, mc);
            
            % B. 运行核心算法: 优化的张量分解 (无 kron 冗余)
            [A1, A2, z_hat] = Spatial_Smoothing_CPD_Optimized(Y, K);
            
            % C. 误差评估与配对 (向量化，无 @ 匿名函数)
            errs = eval_errors_vectorized(z_hat, A2, A1, Frx_c, params, t, K);
            
            err_sq = err_sq + errs;
            if mod(mc, 10) == 0, fprintf('.'); end
        end
        
        % 计算并记录 RMSE
        RES.range(k_idx, p_idx) = sqrt(err_sq(1) / (MC_trials * K));
        RES.vel(k_idx, p_idx)   = sqrt(err_sq(2) / (MC_trials * K));
        RES.aoa(k_idx, p_idx)   = sqrt(err_sq(3) / (MC_trials * K));
        RES.pos(k_idx, p_idx)   = sqrt(err_sq(4) / (MC_trials * K));
        fprintf(' Done\n');
    end
end

%% 3. 绘图部分 (四宫格展示 RMSE)
figure('Color', 'w', 'Position', [100 100 900 700]);
titles = {'(a) AoA', '(b) Range', '(c) Radial velocity', '(d) Position'};
y_labels = {'RMSE (deg)', 'RMSE (m)', 'RMSE (m/s)', 'RMSE (m)'};
fields = {'aoa', 'range', 'vel', 'pos'};

for i = 1:4
    subplot(2, 2, i);
    % K=2 实线，K=4 虚线
    semilogy(P_tx_dBm, RES.(fields{i})(1,:), 'b-o', 'LineWidth', 2, 'MarkerSize', 8); hold on;
    semilogy(P_tx_dBm, RES.(fields{i})(2,:), 'b--s', 'LineWidth', 1.8, 'MarkerSize', 7);
    grid on; 
    xlabel('Transmit power (dBm)', 'FontSize', 11); 
    ylabel(y_labels{i}, 'FontSize', 11); 
    title(titles{i}, 'FontSize', 12);
    if i == 1
        legend('Proposed, K=2', 'Proposed, K=4', 'Location', 'best'); 
    end
    set(gca, 'FontSize', 10);
end
fprintf('\n仿真完成！性能比较图已生成。\n');


%% =========================================================================
%                               子函数模块区
% =========================================================================

%% 模块 A: 信号生成
function [Y, Frx_c] = gen_sig_tensor(p, t, K, snr, mc)
    Yc = zeros(p.R, p.N, p.M); 
    Frx_c = cell(1, K);
    
    for k = 1:K
        vt = sin(t.th(k)) * cos(t.ph(k)); 
        ps = cos(t.th(k));
        a_upa = kron(exp(1j*pi*(0:p.Q-1)'*ps), exp(1j*pi*(0:p.P-1)'*vt));
        
        % 保证每次 Monte Carlo 的噪声独立且可复现
        rng(mc + k*100); 
        Frx = (randn(p.L, p.R) + 1j*randn(p.L, p.R)) / sqrt(2*p.R); 
        Frx_c{k} = Frx;
        
        bk = Frx' * a_upa * (a_upa' * (randn(p.L, 1) + 1j*randn(p.L, 1))/sqrt(2));
        ok = exp(1j * 2*pi * p.Ts * (2*t.v(k)/p.lambda) * (0:p.N-1)');
        gk = exp(-1j * 2*pi * p.delta_f * (2*t.d(k)/p.c0) * (0:p.M-1)');
        
        Yc = Yc + (randn + 1j*randn) * reshape(kron(gk, kron(ok, bk)), [p.R, p.N, p.M]);
    end
    
    sig_p = norm(Yc(:))^2 / numel(Yc);
    Y = Yc + sqrt(sig_p * 10^(-snr/10)) * (randn(size(Yc)) + 1j*randn(size(Yc))) / sqrt(2);
end

%% 模块 B: 核心 CPD 分解 (已优化 kron)
function [A1, A2, z_hat] = Spatial_Smoothing_CPD_Optimized(Y, K)
    [R, N, M] = size(Y);
    Y1_T = reshape(Y, R, N*M).';
    
    L1 = floor(M/2); 
    L2 = M + 1 - L1;
    
    Ys = zeros(L1*N, R*L2);
    for l = 1:L2
        Ys(:, (l-1)*R + 1 : l*R) = Y1_T((l-1)*N + 1 : (l+L1-1)*N, :);
    end
    
    [U, S, V] = svds(Ys, K);
    Xi = pinv(U(1:(L1-1)*N, :)) * U(N+1:L1*N, :);
    [M_mat, Z_diag] = eig(Xi);
    z_hat = diag(Z_diag);
    
    A3 = (z_hat.^(0:M-1)).';
    P_mat = inv(M_mat).';
    
    A2 = zeros(N, K); 
    A1 = zeros(R, K);
    
    for k = 1:K
        % 优化点：使用 reshape 替代 kron 提升速度
        U_M_k = U * M_mat(:, k);
        A2(:, k) = (reshape(U_M_k, N, L1) * conj(A3(1:L1, k))) / norm(A3(1:L1, k))^2;
        
        V_S_P_k = conj(V) * S * P_mat(:, k);
        A1(:, k) = (reshape(V_S_P_k, R, L2) * conj(A3(1:L2, k))) / norm(A3(1:L2, k))^2;
    end
end

%% 模块 C: 误差评估与配对 (向量化实现)
function errs = eval_errors_vectorized(z, A2, A1, Frx, p, t, K)
    errs = zeros(1, 4); % [ed, ev, ea, ep]
    
    % 1. 预计算所有分解出来的距离，将其移出循环，节省计算量
    all_dists_est = abs(angle(z) / (-2*pi*p.delta_f) * p.c0 / 2);
    rem_idx = 1:K; 
    
    for k = 1:K
        % 2. 向量化匹配最接近的距离
        diffs = abs(all_dists_est(rem_idx) - t.d(k));
        [~, min_pos] = min(diffs);
        match_idx = rem_idx(min_pos);
        
        % 3. 提取匹配目标的参数估计值
        dk_e = all_dists_est(match_idx);
        vk_e = (angle(A2(2, match_idx) / A2(1, match_idx)) / (2*pi*p.Ts)) * p.lambda / 2;
        
        % 调用真实 AoA 搜索
        [th_e, ph_e] = GRQ_AoA_Method(A1(:, match_idx), Frx{k}, p.P, p.Q);
        
        % 4. 计算平方误差
        ed = (dk_e - t.d(k))^2;
        ev = (vk_e - t.v(k))^2;
        ea = (rad2deg(th_e) - rad2deg(t.th(k)))^2;
        ep = ed + (t.d(k) * (th_e - t.th(k)))^2; % Position MSE 公式
        
        errs = errs + [ed, ev, ea, ep];
        
        % 5. 从候选池中剔除已匹配索引
        rem_idx(min_pos) = [];
    end
end
