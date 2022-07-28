tfint = tf([1], [1 0 0]);
tfacc = tf([1],[1 0.07 0.005]);
tfgps = tf([0.07 0.005],[1 0.07 0.005]);

t = 0:0.01:10;
w = 1:10;
Am = 0.5;
Sig = Am * t + 10;
acc = Sig + 0.2 * sin(2*pi*50*t);
gps = Am/6 * t.^3 + 10/2* t.^2 + 0.2 * sin(2*pi*5*t);
tr = Am/6 * t.^3 + 10/2* t.^2;
for i=300:400
    gps(i) = 0;
end
res = lsim(tfacc, acc, t) + lsim(tfgps, gps, t);
plot(t, res, t, lsim(tfint, Sig, t), "green", t, tr)
% plot(t, tr-res')