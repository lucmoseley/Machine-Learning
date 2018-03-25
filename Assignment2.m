[numbers, text, everything] = xlsread('48_Industry_Portfolios_daily.csv');

% Maybe bop a 10 year rolling window of what best potfolio is? Optional

predictor_data = numbers(:,2:end);
predictor_names = text(1,2:end);

% Clean data
[row,~] = find(isnan(predictor_data));
row = unique(row);
predictor_data(row,:) = [];

% Calculate full covariance matrix and true sigma^2 of GMVP
C = cov(predictor_data);
true_sigma2GMVP = (ones(1,size(C,1))/C*ones(size(C,1),1))^(-1);

%% K-means
% https://uk.mathworks.com/help/stats/kmeans.html#namevaluepairarguments
%predictor_data = predictor_data';
k = 10;
iterations = 10000;

% 'Start',:'cluster'/'plus'/'sample'/'uniform'
% ^these set different ways of selecting the starting centroid locations

% data is n x p
% idx holds number of cluster each point belongs to: n x 1
% C holds k x p locations of centroids
% sumd holds k x 1 sums over all point-to-centroid dists for each cluster
% D holds n x k distances from every point to every centroid

% Euclidean (square disance)
[idx_eu,C_eu,sumd_eu,D_eu] = kmeans(predictor_data,k,'MaxIter',iterations); 
% Mahanalobis (covariance)
[idx_mah,C_mah,sumd_mah,D_mah] = kmeans(predictor_data,k,'distance','correlation','MaxIter',iterations); 
% Manhattan (abs distance)
[idx_man,C_man,sumd_man,D_man] = kmeans(predictor_data,k,'distance','cityblock','MaxIter',iterations); 
% Cosine (1-cos(angle between vecs)
[idx_cos,C_cos,sumd_cos,D_cos] = kmeans(predictor_data,k,'distance','cosine','MaxIter',iterations); 

% You could (and maybe we will as well) plot W(k) and look for an elbow,
% but we will also look at the gap method: http://web.stanford.edu/~hastie/Papers/gap.pdf
W_eu = W_calc(k,idx_eu,sumd_eu);
W_mah = W_calc(k,idx_man,sumd_man);
W_man = W_calc(k,idx_man,sumd_man);
W_cos = W_calc(k,idx_cos,sumd_cos);

ks = 1:10;
W_ks = zeros(4,10);
for i = 1:10
    [idx_eu,C_eu,sumd_eu,D_eu] = kmeans(predictor_data,i,'MaxIter',iterations);
    [idx_mah,C_mah,sumd_mah,D_mah] = kmeans(predictor_data,i,'distance','correlation','MaxIter',iterations);
    [idx_man,C_man,sumd_man,D_man] = kmeans(predictor_data,i,'distance','cityblock','MaxIter',iterations);
    [idx_cos,C_cos,sumd_cos,D_cos] = kmeans(predictor_data,i,'distance','cosine','MaxIter',iterations);
    W_ks(:,i) = [W_calc(i,idx_eu,sumd_eu);W_calc(i,idx_man,sumd_man);...
        W_calc(i,idx_mah,sumd_mah);W_calc(i,idx_cos,sumd_cos)];
end

names = {'square Euclidian distance','absolute Euclidian distance',...
    'correlation distance','cosine distance'};
figure()
hold on
for i = 1:4
    plot(ks,W_ks(i,:)/max(W_ks(i,:))) %adjust this normalization method
end
legend(names,'fontsize',12)
xlabel('number of clusters','fontsize',14)
ylabel('value within-cluster dispersion','fontsize',14) %scaled value of?
title('Within-cluster dispersion as a function of number of clusters','fontsize',14)

% https://uk.mathworks.com/help/stats/clustering.evaluation.gapevaluation-class.html
% https://uk.mathworks.com/help/stats/evalclusters.html#bt0oocm_sep_shared-clust
% ^MAT gap code
% k-means only does a very specific type :/
% eval might work for not just k-means
% can input diff. criterion where 'gap' is -> maybe even elbow here?
% don't see elbow, but mention possibility of others in report
kmax = 10;
%eva_eu = evalclusters(predictor_data,'kmeans','gap','KList',1:kmax,'Distance','sqEuclidean');
%eva_mah = evalclusters(predictor_data,'kmeans','gap','KList',1:kmax,'Distance','correlation');
%eva_man = evalclusters(predictor_data,'kmeans','gap','KList',1:kmax,'Distance','cityblock');
%eva_cos = evalclusters(predictor_data,'kmeans','gap','KList',1:kmax,'Distance','cosine');

%eva_props = {eva.NumObservations,eva.InspectedK,eva.CriterionValues,eva.OptimalK};

% a = load('eva_mah');
% b = a.eva_mah_props(3);
% b{1}

% gscatter for plotting fun
% Paper on effect of diff dist. measures:
% https://arxiv.org/ftp/arxiv/papers/1405/1405.7471.pdf

%% Dendogram
% note that mahalanobis is not correlation as previously thought, but
% similar b/c it is covariance: (use correlation instead)
% https://uk.mathworks.com/help/stats/linkage.html#d119e463406
linkages = {'average','complete','single'}; %centroid
distances = {'squaredeuclidean','correlation','cityblock','cosine'};

% visual checking: (centroid removed cos got error messages)
% fix x-axis labels so tick nicely
for i = 1:length(linkages);
    for j = 1:length(distances);
        tree = linkage(predictor_data,linkages{i},distances{j});
        %figure(linkages{i}) %distances{j}
        figure(10*i+j)
        dendrogram(tree)
    end
end
% dendogram arguments 'ColorThreshold',cutoff give nice color viewing of
% splits once you've found ideal k

% say this always uses average linkage, even tho Sq(eu) actually does ward
% squared euclidian distance is bad cos doesn't satisfy triangle equality
%(ward looks @ total cluster dissimilarity instead of avg.)
%eva_eu_dend = evalclusters(predictor_data,'linkage','gap','KList',1:kmax,'Distance','sqEuclidean');
%eva_mah_dend = evalclusters(predictor_data,'linkage','gap','KList',1:kmax,'Distance','correlation');
%eva_man_dend = evalclusters(predictor_data,'linkage','gap','KList',1:kmax,'Distance','cityblock');
%eva_cos_dend = evalclusters(predictor_data,'linkage','gap','KList',1:kmax,'Distance','cosine');

% do not need to normalize variables because they are all same type: e.g.
% if one measure was in km and one in mm, then would need to rescale
% evereything to a [0,1] range to weight equally, but in this case all are
% returns (and any diff in returns magnitude for a specific industry is
% relevant to it's clustering position & shouldn't be normalized away!) ->
% only a re-scaling of every data point equally makes sense if you care
% about the number values, which we do for PCA but not here

%% Other clustering techniques

%eva_eu_gmd = evalclusters(predictor_data,'gmdistribution','silhouette','KList',1:kmax,'Distance','sqEuclidean');
%eva_eu_gmd_props = {eva_eu_gmd.NumObservations,eva_eu_gmd.InspectedK,eva_eu_gmd.CriterionValues,eva_eu_gmd.OptimalK};

%% Dendogram
% note that mahalanobis is not correlation as previously thought, but
% similar b/c it is covariance: (use correlation instead)
% https://uk.mathworks.com/help/stats/linkage.html#d119e463406

kmax = 10;

% say this always uses average linkage, even tho Sq(eu) actually does ward
% squared euclidian distance is bad cos doesn't satisfy triangle equality
%(ward looks @ total cluster dissimilarity instead of avg.)
eva_eu_gmd = evalclusters(predictor_data,'gmdistribution','gap','KList',1:kmax,'Distance','sqEuclidean');
%eva_mah_gmd = evalclusters(predictor_data,'gmdistribution','gap','KList',1:kmax,'Distance','correlation');
%eva_man_gmd = evalclusters(predictor_data,'gmdistribution','gap','KList',1:kmax,'Distance','cityblock');
%eva_cos_gmd = evalclusters(predictor_data,'gmdistribution','gap','KList',1:kmax,'Distance','cosine');

eva_eu_gmd_props = {eva_eu_gmd.NumObservations,eva_eu_gmd.InspectedK,eva_eu_gmd.CriterionValues,eva_eu_gmd.OptimalK};
%eva_mah_gmd_props = {eva_mah_gmd.NumObservations,eva_mah_gmd.InspectedK,eva_mah_gmd.CriterionValues,eva_mah_gmd.OptimalK};
%eva_man_gmd_props = {eva_man_gmd.NumObservations,eva_man_gmd.InspectedK,eva_man_gmd.CriterionValues,eva_man_gmd.OptimalK};
%eva_cos_gmd_props = {eva_cos_gmd.NumObservations,eva_cos_gmd.InspectedK,eva_cos_gmd.CriterionValues,eva_cos_gmd.OptimalK};

% https://en.wikipedia.org/wiki/K-means_clustering
% ^search "relation to other..."

%% Plotting 

% K-means
k_sets = {'eva_eu','eva_man','eva_mah','eva_cos'};
k_names = {'square Euclidian distance','absolute Euclidian distance',...
    'correlation distance','cosine distance'};
gap_plot(k_sets,k_names)

% Dendograms
dend_sets = {'eva_eu_dend','eva_man_dend','eva_mah_dend','eva_cos_dend'};
dend_names = {'square Euclidian distance','absolute Euclidian distance',...
    'correlation distance','cosine distance'};
gap_plot(dend_sets,dend_names)

%% PCA

% general pca: the eigenvectors and eigenvalues of the
% covaraince/correlation matrix tell you the components and strengths of
% the various principal components

% explained: tells you how much of the variance is explained by each
% component -- this is all you need for the "scree" plot
% coeff: is the principal components: 1st column = strongest
% latent: tells you variance around each component

predictor_data = predictor_data';

% precautionary normalization for if results don't match to k-means:
%{
for i = 1:48
    predictor_data(:,i) = predictor_data(:,i) - mean(predictor_data(:,i));
    predictor_data(:,i) = predictor_data(:,i)/std(predictor_data(:,i));
end
%}
% we do not normalization because the differences in the scale of returns
% size is important. If we weren't measuring returns for all then maybe we
% would, but we are

stds = zeros(1,48);
for i = 1:48
    stds(i) = std(predictor_data(:,i));
end

[coeff,score,latent,tsquared,explained,mu] = pca(predictor_data);

x=[1:size(explained,1)]';
figure()
hold on
plot(x,explained)
plot(x,cumsum(explained))
ylabel('percentage of variance explained','fontsize',14)
xlabel('principal component number','fontsize',14)
title('Scree plot showing importances of principal component egienvalues','fontsize',14)
legend({'captured variance per component','cumulative captured variance'},'fontsize',12)

% plot amount of each industry in princiapl component & first 5 comps.
%for i = 1:48
%    coeff(:,i) = coeff(:,i)/sum(coeff(:,i));
%end
figure()
hold on
plot(x,coeff(:,1))
plot(x,sum(coeff(:,1:5),2))
ylabel('proportional size','fontsize',14)
xlabel('industry','fontsize',14)
title('Proportional importance of each industry in representing the market','fontsize',14)
legend({'first principal component','sum over first 5 principal components'},'fontsize',12)
xticks(x)
xtickangle(45)
xticklabels(industries)
xlim([1,48])

% below shows that, even without normalization, stds do not fully predict
% principal components -- maybe do PCA with and without normalization to 
% show effects due to variation AANNDD those due to alignment with market

% plot relative standard deviations of each industry for comparison to PCs
figure()
plot(x,stds)
ylabel('proportional size of standard deviation','fontsize',14)
xlabel('industry','fontsize',14)
title('Proportional standard deviations of each industry within the market','fontsize',14)
%legend({'first principal component','sum over first 5 principal components'},'fontsize',12)
xticks(x)
xtickangle(45)
xticklabels(industries)
xlim([1,48])

% most and least important industries
pc1 = coeff(:,1);
pc1_max2 = pc1(pc1~=max(pc1));
pc1_min2 = pc1(pc1~=min(pc1));
pc5 = abs(sum(coeff(:,1:5),2));
pc5_max2 = pc5(pc5~=max(pc5));
pc5_min2 = pc5(pc5~=min(pc5));
most_pc1 = [industries(pc1==max(pc1)),industries(pc1==max(pc1_max2)),industries(pc1==max(pc1_max2(pc1_max2~=max(pc1_max2))))];
most_pc5 = [industries(pc5==max(pc5)),industries(pc5==max(pc5_max2)),industries(pc5==max(pc5_max2(pc5_max2~=max(pc5_max2))))];
least_pc1 = [industries(pc1==min(pc1)),industries(pc1==min(pc1_min2)),industries(pc1==min(pc1_min2(pc1_min2~=min(pc1_min2))))];
least_pc5 = [industries(pc5==min(pc5)),industries(pc5==min(pc5_min2)),industries(pc5==min(pc5_min2(pc5_min2~=min(pc5_min2))))];

std_max2 = stds(stds~=max(stds));
std_min2 = stds(stds~=min(stds));
most_std = [industries(stds==max(stds)),industries(stds==max(std_max2)),industries(stds==max(std_max2(std_max2~=max(std_max2))))];
least_std = [industries(stds==min(stds)),industries(stds==min(std_min2)),industries(stds==min(std_min2(std_min2~=min(std_min2))))];
