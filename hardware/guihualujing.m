%三次多项式插值法
clear;
clc;
q_array=[0,30,90,30,0];%指定起止位置
t_array=[0,3,6,12,14];%指定起止时间
v_array=[0,20,30,-15,0];%指定起止速度
t=[t_array(1)];q=[q_array(1)];v=[v_array(1)];a=[0];%初始状态
for i=1:1:length(q_array)-1%每一段规划的时间
     a0=q_array(i);
     a1=v_array(i);
     a2=(3/(t_array(i+1)-t_array(i))^2)*(q_array(i+1)-q_array(i))-(1/(t_array(i+1)-t_array(i)))*(2*v_array(i)+v_array(i+1));
     a3=(2/(t_array(i+1)-t_array(i))^3)*(q_array(i)-q_array(i+1))+(1/(t_array(i+1)-t_array(i))^2)*(v_array(i)+v_array(i+1));
     ti=t_array(i)+0.001:0.001:t_array(i+1);
     qi=a0+a1*(ti-t_array(i))+a2*(ti-t_array(i)).^2+a3*(ti-t_array(i)).^3;
     vi=a1+2*a2*(ti-t_array(i))+3*a3*(ti-t_array(i)).^2;
     ai=2*a2+6*a3*(ti-t_array(i));
     t=[t,ti];q=[q,qi];v=[v,vi];a=[a,ai];
end
subplot(3,1,1),plot(t,q,'r'),xlabel('t/s'),ylabel('p/m');hold on; plot(t_array,q_array,'o','color','r'),grid on;
subplot(3,1,2),plot(t,v,'b'),xlabel('t/s'),ylabel('v/(m/s)');hold on;plot(t_array,v_array,'*','color','r'),grid on;
subplot(3,1,3),plot(t,a,'g'),xlabel('t/s'),ylabel('a/(m/s^2)');hold on;
% 指定文件夹保存图片
filepath=pwd;           %保存当前工作目录
cd('E:\SeniorYearUp\Final\paper\image\backup')                %把当前工作目录切换到图片存储文件夹
print(gcf,'-djpeg','E:\SeniorYearUp\Final\paper\image\backup\san.jpeg'); %将图片保存为jpg格式，
cd(filepath)            %切回原工作目录


%五次多项式插值法
clear;
clc;
q_array=[0,30,90,30,0];%指定起止位置
t_array=[0,3,6,12,14];%指定起止时间
v_array=[0,20,30,-15,0];%指定起止速度
a_array=[0,10,15,-15,0];%指定起止加速度
t=[t_array(1)];q=[q_array(1)];v=[v_array(1)];a=[a_array(1)];%初始状态
for i=1:1:length(q_array)-1%每一段规划的时间
     T=t_array(i+1)-t_array(i);
     a0=q_array(i);
     a1=v_array(i);
     a2=a_array(i)/2;
     a3=(20*q_array(i+1)-20*q_array(i)-(8*v_array(i+1)+12*v_array(i))*T-(3*a_array(i)-a_array(i+1))*T^2)/(2*T^3);
     a4=(30*q_array(i)-30*q_array(i+1)+(14*v_array(i+1)+16*v_array(i))*T+(3*a_array(i)-2*a_array(i+1))*T^2)/(2*T^4);
     a5=(12*q_array(i+1)-12*q_array(i)-(6*v_array(i+1)+6*v_array(i))*T-(a_array(i)-a_array(i+1))*T^2)/(2*T^5);%计算五次多项式系数 
     ti=t_array(i):0.001:t_array(i+1);
     qi=a0+a1*(ti-t_array(i))+a2*(ti-t_array(i)).^2+a3*(ti-t_array(i)).^3+a4*(ti-t_array(i)).^4+a5*(ti-t_array(i)).^5;
     vi=a1+2*a2*(ti-t_array(i))+3*a3*(ti-t_array(i)).^2+4*a4*(ti-t_array(i)).^3+5*a5*(ti-t_array(i)).^4;
     ai=2*a2+6*a3*(ti-t_array(i))+12*a4*(ti-t_array(i)).^2+20*a5*(ti-t_array(i)).^3;
     t=[t,ti(2:end)];q=[q,qi(2:end)];v=[v,vi(2:end)];a=[a,ai(2:end)];
end
subplot(3,1,1),plot(t,q,'r'),xlabel('t/s'),ylabel('p/m');hold on; plot(t_array,q_array,'o','color','r'),grid on;
subplot(3,1,2),plot(t,v,'b'),xlabel('t/s'),ylabel('v/(m/s)');hold on;plot(t_array,v_array,'*','color','r'),grid on;
subplot(3,1,3),plot(t,a,'g'),xlabel('t/s'),ylabel('a/(m/s^2)');hold on;plot(t_array,a_array,'^','color','r'),grid on;
% 指定文件夹保存图片
filepath=pwd;           %保存当前工作目录
cd('E:\SeniorYearUp\Final\paper\image\backup')                %把当前工作目录切换到图片存储文件夹
print(gcf,'-djpeg','E:\SeniorYearUp\Final\paper\image\backup\wu.jpeg'); %将图片保存为jpg格式，
cd(filepath)            %切回原工作目录





% Modified DH
% ABB robot
% lujingguihua
clear;
clc;
% %机器人建模
th(1) = 0; d(1) = 0; a(1) = 0; alp(1) = 0;
th(2) = 0; d(2) = 0; a(2) = 3.20; alp(2) = pi/2;   
th(3) = 0; d(3) = 0; a(3) = 9.70; alp(3) = 0;
th(4) = 0; d(4) = 8.80; a(4) = 2; alp(4) = pi/2;
th(5) = 0; d(5) = 0; a(5) = 0; alp(5) = -pi/2;
th(6) = 0; d(6) = 0; a(6) = 0; alp(6) = pi/2;
% DH parameters  th     d    a    alpha  sigma
L1 = Link([th(1), d(1), a(1), alp(1), 0], 'modified');
L2 = Link([th(2), d(2), a(2), alp(2), 0], 'modified');
L3 = Link([th(3), d(3), a(3), alp(3), 0], 'modified');
L4 = Link([th(4), d(4), a(4), alp(4), 0], 'modified');
L5 = Link([th(5), d(5), a(5), alp(5), 0], 'modified');
L6 = Link([th(6), d(6), a(6), alp(6), 0], 'modified');
robot = SerialLink([L1, L2, L3, L4, L5, L6]); %SerialLink 类函数
robot.name='Robot-6-dof';
robot.display(); %显示D-H表

%轨迹规划参数设置
init_ang = [pi/6,0, 3*pi/7,pi/3, 0, 0];
targ_ang = [pi/2,pi/4,0,0, -pi/3, pi/6];
T =(0:0.1:10);
%关节空间轨迹规划方法
[q,qd,qdd] = jtraj(init_ang,targ_ang,T); %直接得到角度、角速度、角加速度的的序列

%%显示
figure(1);
%动画显示
subplot(1,2,1); 
title('动画过程');
robot.plot(q);
% 轨迹显示
t=robot.fkine(q);%运动学正解
rpy=tr2rpy(t);  %t中提取位置（xyz）
subplot(1,2,2);
plot2(rpy);
xlabel('X/mm'),ylabel('Y/mm'),zlabel('Z/mm');hold on
title('空间轨迹');
text(rpy(1,1),rpy(1,2),rpy(1,3),'A点');
text(rpy(51,1),rpy(51,2),rpy(51,3),'B点');
% 指定文件夹保存图片
filepath=pwd;           %保存当前工作目录
cd('E:\SeniorYearUp\Final\paper\image\backup')                %把当前工作目录切换到图片存储文件夹
print(gcf,'-djpeg','E:\SeniorYearUp\Final\paper\image\backup\1.jpeg'); %将图片保存为jpg格式，
cd(filepath)            %切回原工作目录

%单个关节的位置title('关节1位置');
figure(2);
subplot(3,2,1);
plot(T,q(:,1));
xlabel('t/s'),ylabel('\theta_1/rad');hold on
subplot(3,2,2);
plot(T,q(:,2));
xlabel('t/s'),ylabel('\theta_2/rad');hold on
subplot(3,2,3);
plot(T,q(:,3));
xlabel('t/s'),ylabel('\theta_3/rad');hold on
subplot(3,2,4);
plot(T,q(:,4));
xlabel('t/s'),ylabel('\theta_4/rad');hold on
subplot(3,2,5);
plot(T,q(:,5));
xlabel('t/s'),ylabel('\theta_5/rad');hold on
subplot(3,2,6);
plot(T,q(:,6));
xlabel('t/s'),ylabel('\theta_6/rad');hold on
% 指定文件夹保存图片
filepath=pwd;           %保存当前工作目录
cd('E:\SeniorYearUp\Final\paper\image\backup')                %把当前工作目录切换到图片存储文件夹
print(gcf,'-djpeg','E:\SeniorYearUp\Final\paper\image\backup\2.jpeg'); %将图片保存为jpg格式，
cd(filepath)            %切回原工作目录

%单个关节的速度
figure(3);
subplot(3,2,1);
plot(T,qd(:,1));
xlabel('t/s'),ylabel('\Omega_1/rad');hold on
subplot(3,2,2);
plot(T,qd(:,2));
xlabel('t/s'),ylabel('\Omega_2/rad');hold on
subplot(3,2,3);
plot(T,qd(:,3));
xlabel('t/s'),ylabel('\Omega_3/rad');hold on
subplot(3,2,4);
plot(T,qd(:,4));
xlabel('t/s'),ylabel('\Omega_4/rad');hold on
subplot(3,2,5);
plot(T,qd(:,5));
xlabel('t/s'),ylabel('\Omega_5/rad');hold on
subplot(3,2,6);
plot(T,qd(:,6));
xlabel('t/s'),ylabel('\Omega_6/rad');hold on
% 指定文件夹保存图片
filepath=pwd;           %保存当前工作目录
cd('E:\SeniorYearUp\Final\paper\image\backup')                
%把当前工作目录切换到图片存储文件夹
print(gcf,'-djpeg','E:\SeniorYearUp\Final\paper\image\backup\3.jpeg'); %将图片保存为jpg格式，
cd(filepath)            %切回原工作目录

%单个关节的加速度
figure(4);
subplot(3,2,1);
plot(T,qdd(:,1));
xlabel('t/s'),ylabel('\alpha_1/rad');hold on
subplot(3,2,2);
plot(T,qdd(:,2));
xlabel('t/s'),ylabel('\alpha_2/rad');hold on
subplot(3,2,3);
plot(T,qdd(:,3));
xlabel('t/s'),ylabel('\alpha_3/rad');hold on
subplot(3,2,4);
plot(T,qdd(:,4));
xlabel('t/s'),ylabel('\alpha_4/rad');hold on;
subplot(3,2,5);
plot(T,qdd(:,5));
xlabel('t/s'),ylabel('\alpha_5/rad');hold on
subplot(3,2,6);
plot(T,qdd(:,6));
xlabel('t/s'),ylabel('\alpha_6/rad');hold on
% 指定文件夹保存图片
filepath=pwd;           %保存当前工作目录
cd('E:\SeniorYearUp\Final\paper\image\backup')                %把当前工作目录切换到图片存储文件夹
print(gcf,'-djpeg','E:\SeniorYearUp\Final\paper\image\backup\4.jpeg'); %将图片保存为jpg格式，
cd(filepath)            %切回原工作目录
