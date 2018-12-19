/*
Copyright (C) 2001  Kyle Siegrist, Dawn Duehring

This program is free software; you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the Free
Software Foundation; either version 2 of the License, or (at your option)
any later version.

This program is distributed in the hope that it will be useful, but without
any warranty; without even the implied warranty of merchantability or
fitness for a particular purpose. See the GNU General Public License for
more details. You should have received a copy of the GNU General Public
License along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
*/
package statistics.distributions;

/**The distribution of the order statistic of a specified order from a
random sample of a specified size from a specified sampling distribution*/
public class OrderStatisticDistribution extends Distribution{
	Distribution dist;
	int sampleSize, order;

	/**General constructor: creates a new order statistic distribution
	corresponding to a specified sampling distribution, sample size, and
	order*/
	public OrderStatisticDistribution(Distribution d, int n, int k){
		setParameters(d, n, k);
	}

	/**Set the parameters: the sampling distribution, sample size, and order*/
	public void setParameters(Distribution d, int n, int k){
		//Correct for invalid parameters
		if (n < 1) n = 1;
		if (k < 1) k = 1; else if (k > n) k = n;
		//Assign parameters
		dist = d;
		sampleSize = n;
		order = k;
		int t = dist.getType();
		Domain domain = dist.getDomain();
		if (t == DISCRETE) super.setParameters(domain.getLowerValue(), domain.getUpperValue(), domain.getWidth(), t);
		else super.setParameters(domain.getLowerBound(), domain.getUpperBound(), domain.getWidth(), t);
	}

	/**Density function*/
	public double getDensity(double x){
		double p = dist.getCDF(x);
		if (dist.getType() == DISCRETE) return getCDF(x) - getCDF(x - getDomain().getWidth());
		else return order * comb(sampleSize, order) * Math.pow(p, order - 1) * Math.pow(1 - p, sampleSize - order) * dist.getDensity(x);
	}

	/**Cumulative distribution function*/
	public double getCDF(double x){
		double sum = 0;
		double p = dist.getCDF(x);
		for (int j = order; j <= sampleSize; j++) sum = sum + comb(sampleSize, j) * Math.pow(p, j) * Math.pow(1 - p, sampleSize - j);
		return sum;
	}
}

