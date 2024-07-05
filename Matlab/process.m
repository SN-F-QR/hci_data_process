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

% use cell to arrange all data group
d_groups = {d_no, d_arr, d_sw, d_hl};
for i = 1:length(d_groups)
    d_groups{i}(:,2) = 8 - d_groups{i}(:,2);
    d_groups{i}(:,4) = 8 - d_groups{i}(:,4);
end

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

% % 计算每个问题的平均值和标准差
% res = zeros(4,numel(fields_arr));
% res_std = zeros(4, howMuchQuesOneGroup);
% res_diff_p = zeros(1, howMuchQuesOneGroup);
% x_axis = strings(1, numel(fields_arr));
% for k=1:numel(fields_arr)
%     name = string(fields_arr(k));
%     names = split(name, '_');
%     name = names(1);
%     disp(name);
%     x_axis(k) = name;
% 
%     %showNormality(s_2d.(fields_2d{k}));
%     %showNormality(s_3d.(fields_3d{k}));
% 
%     for m=1:length(d_groups)
%         res(4*k+m-4) = mean(d_groups{m}(:,k));
%         res_std(4*k+m-4) = std(d_groups{m}(:,k), 1);
%     end
% 
%     if (res(4*k-3) ~= 0)
%         % res_diff_p(i) = signrank(d_mean(1:14,i), d_mean(15:28,i));
%         res_diff_p(k) = friedman([d_groups{1}(:,k),d_groups{2}(:,k), d_groups{3}(:,k), d_groups{4}(:,k)], 1, "off");
%     else
%         % res_diff_p(i) = signrank(d_mean(1:14,i), d_mean(15:28,i));
%         res_diff_p(k) = friedman([d_groups{2}(:,k), d_groups{3}(:,k), d_groups{4}(:,k)], 1, "off");
%     end
% 
%     % res(2*k-1) = mean(s_2d.(fields_2d{k}));
%     % res(2*k) = mean(s_3d.(fields_3d{k}));
%     % 
%     % res_std(2*k-1) = std(s_2d.(fields_2d{k}), 1);
%     % res_std(2*k) = std(s_3d.(fields_3d{k}), 1);
% 
%     % Wilcoxon signed rank test
%     % res_diff_p(k) = signrank(s_2d.(fields_2d{k}), s_3d.(fields_3d{k}));
% 
% end
% 
% % post-hoc处理
% post_hoc = cell(100,1);
% n = 1;
% for k = 1:15  % ques group count
%     if (res(1,k) ~= 0)
%         start = 1;
%     else
%         start = 2;
%     end
%     if (res_diff_p(k) < 0.06)
%         for i = start:groupNum
%             for j = i+1:groupNum
%                 p = signrank(d_groups{i}(:,k), d_groups{j}(:,k));
%                 post_hoc{n} = {k,i,j,p};
%                 n = n + 1;
%             end
%         end
%     end
% end

% 仅计算每个sub-scale的总平均分和标准差

res = zeros(4, 6);
res_std = zeros(4, 6);
res_diff_p = zeros(1, 6);
d_mean = zeros(56, 6);
x_axis = strings(1, 6);
last_name = "SA";
i = 1;
j = 1;
for k=1:(numel(fields_arr)+1)
    if (k < numel(fields_arr)+1)
        name = string(fields_arr(k));
        names = split(name, '_');
        group_name = names(1);
        if any(group_name == ["UE1", "UE2", "UE3", "UE4"])
            group_name = "PQ";
        elseif any(group_name == ["UE5", "UE6", "UE7", "UE8"])
            group_name = "HQ";
        elseif (group_name == "TR2")
            group_name = "IT";
        end
    else
        group_name = "End";
    end


    group_name = regexprep(group_name, '[^a-zA-Z]', '');  % clean num in name
    % calculate mean when the ques group changes
    if (group_name ~= last_name)  
        x_axis(i) = last_name;
        last_name = group_name;
        % mean for each experiment group
        for m=1:length(d_groups)
            d_mean(14*m-13:14*m, i) = mean(d_groups{m}(:,j:k-1), 2);
            res(m,i) = mean(d_mean(14*m-13:14*m, i));
            res_std(m,i) = std(d_mean(14*m-13:14*m, i), 1);
            % if any(i == [1,2,4])
            %     % res_diff_p(i) = signrank(d_mean(1:14,i), d_mean(15:28,i));
            %     res_diff_p(i) = friedman([d_mean(1:14,i),d_mean(15:28,i), d_mean(29:42,i), d_mean(43:56,i)], 1, "on");
            % else
            %     % res_diff_p(i) = signrank(d_mean(1:14,i), d_mean(15:28,i));
            %     res_diff_p(i) = friedman([d_mean(15:28,i), d_mean(29:42,i), d_mean(43:56,i)], 1, "on");
            % end
        end

        if any(i == [1,2,4])
            % res_diff_p(i) = signrank(d_mean(1:14,i), d_mean(15:28,i));
            res_diff_p(i) = friedman([d_mean(1:14,i),d_mean(15:28,i), d_mean(29:42,i), d_mean(43:56,i)], 1, "on");
        else
            % res_diff_p(i) = signrank(d_mean(1:14,i), d_mean(15:28,i));
            res_diff_p(i) = friedman([d_mean(15:28,i), d_mean(29:42,i), d_mean(43:56,i)], 1, "on");
        end
        % d_mean(1:14,i) = mean(d_no(:,j:k-1), 2);
        % d_mean(15:28,i) = mean(d_arr(:,j:k-1), 2);
        % d_mean(29:43,i) = mean(d_sw(:,j:k-1), 2);
        % d_mean(44:58,i) = mean(d_hl(:,j:k-1), 2);
        % res(1,i) = mean(d_mean(1:14, i));
        % res(2,i) = mean(d_mean(15:28, i));
        % res(3,i) = mean(d_mean(29:43, i));
        % res(4,i) = mean(d_mean(44:58, i));
        % res_std(1,i) = std(d_mean(1:14, i), 1);
        % res_std(2,i) = std(d_mean(15:28, i), 1);
        % res_diff_p(i) = signrank(d_mean(1:14,i), d_mean(15:28,i));

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
% types = categorical(names(types));

% deal with experiment groups
group = ones(size(type, 1), 1) * (1:groupNum);
group = group(:);


% post-hoc处理
post_hoc = cell(100,1);
n = 1;
for k = 1:6  % ques group count
    if any(k == [1,2,4])
        start = 1;
    else
        start = 2;
    end
    if (res_diff_p(k) < 0.06)
        for i = start:groupNum
            for j = i+1:groupNum
                p = signrank(d_mean(i*14-13:i*14,k), d_mean(j*14-13:j*14,k));
                post_hoc{n} = {k,i,j,p};
                n = n + 1;
            end
        end
    end
end

            
        

figure;
% 分组柱状图
graph_b = bar(x_axis, res, 'EdgeColor', 'none');
set(gca, 'Ygrid', 'on');
set(gca, 'ytick', (1:7));
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
        'VerticalAlignment','bottom', 'FontSize', 10);
    graph_b(i).FaceColor = RGB(i);
end
hold off;

% xtips_1 = graph_b(1).XEndPoints;
% xtips_2 = xtips;
tmp_k = 0;
for i = 1:length(post_hoc) 
    if isempty(post_hoc{i})
        break
    end
    [exp_group, left, right, p] = post_hoc{i}{:};
    xtip1 = graph_b(left).XEndPoints(exp_group);
    xtip2 = graph_b(right).XEndPoints(exp_group);
    if (exp_group ~= tmp_k)
        basic_height = 6.5;
        tmp_k = exp_group;
        n = 0;
    % elseif(p <=0.05)
        % basic_height = basic_height + 0.5;
    end
    if (p <= 0.001)
        sigline(3, [xtip1, xtip2], [], basic_height + 0.5*n);
        n = n + 1;
    elseif (p <= 0.01)
        sigline(2, [xtip1, xtip2], [], basic_height + 0.5*n);
        n = n + 1;
    elseif (p <= 0.05)
        % sigline(1, [xtips_1(i), xtips_2(i)], [], 5);
        sigline(1, [xtip1, xtip2], [], basic_height + 0.5*n);
        n = n + 1;
    end
end

    
% 箱线图
% boxchart(types, data, 'GroupByColor', group);
ylim([1,8]);
legend(["NoRS", "Arr.", "Swap", "High."]);


function showNormality(x)
    [isNormal,pvalue,wvalue] = swtest(x);
    disp("Normality is " + ~isNormal + " with p = " + pvalue + " w = " + wvalue);
end