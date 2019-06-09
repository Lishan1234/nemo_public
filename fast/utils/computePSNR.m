function psnr_value = computePSNR( X0, X )
% computePSNR(X0, X) computes the peak signal-to-noise ratio defined as:
% PSNR(X0, X) = 20 * log10(max(|X0|)) - 10 * log10(MSE(X0, X))
% ------------------------input-------------------------------------------
% X0, X:        two matrices with the same size.
% ------------------------output------------------------------------------
% psnr_value:   peak signal-to-noise ratio between X0 and X.
X = double(X);
X0 = double(X0);
psnr_value = 20 * log10(255) - 10 * log10(sum((X0(:) -  X(:)).^2) / numel(X));
