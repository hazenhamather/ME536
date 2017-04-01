function [posxy] = computeKinematics(q)
%function [posxy] = computeKinematics(q)
%This function computes the x and y position of the end effector when given
%the three joint angles
%Input
%q - three angles of the arm joints
%Outout
%posxy - x and y position of the end effector as a vector

l1 = 1; %m
l2 = 1; %m
l3 = 1; %m
posxy = [l1*cos(q(1))+l2*cos(q(1)+q(2))+l3*cos(q(1)+q(2)+q(3));...
    l1*sin(q(1))+l2*sin(q(1)+q(2))+l3*sin(q(1)+q(2)+q(3))];