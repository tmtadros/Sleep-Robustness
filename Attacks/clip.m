function [xclp] = clip(x,xmin,xmax)

xclp = x;
xclp(xclp<xmin) = xmin;
xclp(xclp>xmax) = xmax;

end
