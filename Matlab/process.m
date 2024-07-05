clc;
clear;
addpath(genpath(userpath))
addpath("~/Developer/Exp_Result/")
path = "~/Developer/Exp_Result/VR Rec Questionnaire.csv";

usefulDataStartFrom = 5;  % comparable data
howMuchQuesOneGroup = 15;
groupNum = 4;  % how much experiment group

% color setting
% RGB = ["#8ECFC9", "#FA7F6F"];
RGB = ["#ADD8E6", "#FFDAB9", "#E6E6FA", "#F1A7B5"];
RGB_Gray = "#9E9E9E";

ques = readtable(path);
s_no = table2struct(ques(:,5:13), "ToScalar",true);
s_arr = table2struct(ques(:,14:28), "ToScalar",true);
s_sw = table2struct(ques(:,29:43), "ToScalar",true);
s_hl = table2struct(ques(:,44:58), "ToScalar",true);
data = ques(:, 5:58);
d_no = table2array(ques(:,5:13));
d_no = [d_no(:,1:8),zeros(14,4),d_no(:,9),zeros(14,2)];
d_arr = table2array(ques(:,14:28));
d_sw = table2array(ques(:,29:43));
d_hl = table2array(ques(:,44:58));

% 转置table以匹配boxchart的要求
% row_ques = rows2vars(ques);
% row_ques = row_ques(5:46, :);
% row_ques = renamevars(row_ques, "OriginalVariableNames", "Type");
% num_people = length(row_ques{1,:}) - 1; % 计算实验参与人数

data = table2array(data);
data = data(:); % convert to one column

type = ones(height(ques), 1) * (1:howMuchQuesOneGroup);
type = type(:); % convert to one column
types = repmat(type, groupNum, 1); % repeat to match exp group

% deal with the name

fields_arr = fieldnames(s_arr);
fields_sw = fieldnames(s_sw);

% 计算每个问题的平均值和标准差
% res = zeros(2,numel(fields_2d));
% res_std = zeros(2, howMuchQuesOneGroup);
% res_diff_p = zeros(1, howMuchQuesOneGroup);
% x_axis = strings(1, numel(fields_2d));
% for k=1:numel(fields_2d)
%     name = string(fields_2d(k));
%     names = split(name, '_');
%     name = names(2) + k;
%     disp(name);
%     x_axis(k) = name;
% 
%     %showNormality(s_2d.(fields_2d{k}));
%     %showNormality(s_3d.(fields_3d{k}));
% 
%     res(2*k-1) = mean(s_2d.(fields_2d{k}));
%     res(2*k) = mean(s_3d.(fields_3d{k}));
% 
%     res_std(2*k-1) = std(s_2d.(fields_2d{k}), 1);
%     res_std(2*k) = std(s_3d.(fields_3d{k}), 1);
% 
%     % Wilcoxon signed rank test
%     res_diff_p(k) = signrank(s_2d.(fields_2d{k}), s_3d.(fields_3d{k}));
% 
% end

% 仅计算每个sub-scale的总平均分和标准差

res = zeros(2, 6);
res_std = zeros(2, 6);
res_diff_p = zeros(1, 6);
d_mean = zeros(28, 6);
x_axis = strings(1, 6);
last_name = "Efficiency";
i = 1;
j = 1;
for k=1:numel(fields_2d)
    name = string(fields_2d(k));
    names = split(name, '_');
    if (names(2) == "Natural")
        names(2) = "Immersion";
    elseif (k==numel(fields_2d))
        names(2) = "End";
        k = k + 1;
    end
    if (names(2) ~= last_name)
        x_axis(i) = last_name;
        last_name = names(2);
        d_mean(1:14,i) = mean(d_2d(:,j:k-1), 2);
        d_mean(15:28,i) = mean(d_3d(:,j:k-1), 2);
        res(1,i) = mean(d_mean(1:14, i));
        res(2,i) = mean(d_mean(15:28, i));
        res_std(1,i) = std(d_mean(1:14, i), 1);
        res_std(2,i) = std(d_mean(15:28, i), 1);
        res_diff_p(i) = signrank(d_mean(1:14,i), d_mean(15:28,i));

        i = i+1;
        j = k;
    end

    %disp(name);

    %showNormality(s_2d.(fields_2d{k}));
    %showNormality(s_3d.(fields_3d{k}));

    % Wilcoxon signed rank test
    

end

% assign string name to types
names = x_axis';
types = categorical(names(types));

% deal with experiment groups
group = ones(size(type, 1), 1) * (1:groupNum);
group = group(:);


figure;
% 分组柱状图
graph_b = bar(x_axis, res, 'EdgeColor', 'none');
set(gca, 'Ygrid', 'on');
set(gca, 'ytick', (1:5));
% set(gca, 'xtick', []);

hold on;
% add mean text and error bar(std)
for i = 1:groupNum
    xtips = graph_b(i).XEndPoints;
    ytips = graph_b(i).YEndPoints;
    labels = string(round(graph_b(i).YData, 2));
    e = errorbar(xtips, ytips, res_std(i,:), 'LineStyle', 'none',...
    'LineWidth', 2);
    e.Color = RGB_Gray;
    text(xtips,ytips,labels,'HorizontalAlignment','center', ...
        'VerticalAlignment','bottom');
    graph_b(i).FaceColor = RGB(i);
end
hold off;

xtips_1 = graph_b(1).XEndPoints;
xtips_2 = xtips;
for i = 1:6 % howMuchQuesOneGroup
    if (res_diff_p(i) <= 0.001)
        sigline(3, [xtips_1(i), xtips_2(i)], [], 5);
    elseif (res_diff_p(i) <= 0.01)
        sigline(2, [xtips_1(i), xtips_2(i)], [], 5);
    elseif (res_diff_p(i) <= 0.05)
        sigline(1, [xtips_1(i), xtips_2(i)], [], 5);
    end
end

    
% 箱线图
% boxchart(types, data, 'GroupByColor', group);
ylim([1,6]);
legend(["2D store", "3D store"]);


function showNormality(x)
    [isNormal,pvalue,wvalue] = swtest(x);
    disp("Normality is " + ~isNormal + " with p = " + pvalue + " w = " + wvalue);
end