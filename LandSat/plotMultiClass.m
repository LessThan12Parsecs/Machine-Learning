function [work] = plotMultiClass(X,Y,classes)
%plots every class in a multivariate dataset as a different color. 
colors = ['r','y','k','b','g','c'];
legends = ['RedSoil','CottonCrop','GreySoil','DampGreySoil','SoilVegetation','VeryDampGrey'];
for i=1:classes
    y_plot = (Y==i);
    res1 = X(:,4).*y_plot;
    res2 = X(:,3).*y_plot;
    res1(res1==0)=[];
    res2(res2==0)=[];
    m = scatter(res1,res2,colors(i));
    hold on;
end
    legend

end

