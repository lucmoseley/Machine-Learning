function [ ] = gap_plot( eval_obj_names, names )
%Make plots from evaluation objects

len = length(eval_obj_names);
ks = zeros(len,10);
gap_vals = zeros(len,10);
optim_ks = zeros(len,1);

for i = 1:len
    struct = load(eval_obj_names{i});
    cell = struct2cell(struct);
    set = cell{1};
    %num_obs = cell2mat(set(1));
    ks(i,:) = cell2mat(set(2));
    gap_vals(i,:) = cell2mat(set(3));
    optim_ks(i) = cell2mat(set(4));
end

figure()
hold on
for i = 1:len
    plot(ks(i,:),gap_vals(i,:))
    %vline(optim_ks(i),{'optimal clustering for',names{i}})
end
legend(names,'fontsize',12)
xlabel('number of clusters','fontsize',14)
ylabel('value of the gap clustering criterion','fontsize',14)
title('Gap optimal clustering evaluation for k-means','fontsize',14)
% change title for dendos

end
