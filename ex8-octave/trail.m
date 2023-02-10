load('ex8data1.mat');
#plot(X(:, 1),y, 'bx');
#axis([0 30 0 30]);
#xlabel('Latency (ms)');
#ylabel('Throughput (mb/s)');

[mu sigma2] = estimateGaussian(X(:,1));
p = multivariateGaussian(X(:,1), mu, sigma2);

z=(1/(2*pi*sigma2^2))*exp(-((X(:,1)-mu).^2)./(2*sigma2^2));
q=0:0.1:30;
z2=(1/(2*pi*sigma2^2))*exp(-((q-mu).^2)./(2*sigma2^2));

plot(q,z2)
hold all

plot(X(:,1),z,'*');
xlabel('Latency (ms)');


