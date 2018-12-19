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

/**This class applies a location-scale tranformation to a given distribution. In terms of
 * the corresponding random variable X, the transformation is Y = a + bX*/
public class LocationScaleDistribution extends Distribution{
	private Distribution dist;
	private double location, scale;

	/**This general constructor creates a new location-scale transformation on
	a given distribuiton with given location and scale parameters*/
	public LocationScaleDistribution(Distribution d, double a, double b){
		setParameters(d, a, b);
	}

	/**This method sets the parameters: the distribution and the location and
	scale parameters*/
	public void setParameters(Distribution d, double a, double b){
		dist = d; location = a; scale = b;
		Domain domain = dist.getDomain();
		double l, u, w = domain.getWidth();
		int t = dist.getType();
		if (t == DISCRETE){
			l = domain.getLowerValue(); u = domain.getUpperValue();
		}
		else{
			l = domain.getLowerBound(); u = domain.getUpperBound();
		}
		if (scale == 0) super.setParameters(location, location, 1, DISCRETE);
		else if (scale < 0) super.setParameters(location + scale * u, location + scale * l, -scale * w, t);
		else super.setParameters(location + scale * l, location + scale * u, scale * w, t);
	}

	/**This method defines the getDensity function*/
	public double getDensity(double x){
		if (scale == 0){
			if (x == location) return 1;
			else return 0;
		}
		else return dist.getDensity((x - location) / scale);
	}

	/**This method returns the maximum value of the getDensity function*/
	public double getMaxDensity(){
		return dist.getMaxDensity();
	}

	/**This mtehod returns the mean*/
	public double getMean(){
		return location + scale * dist.getMean();
	}

	/**This method returns the variance*/
	public double getVariance(){
		return (scale * scale) * dist.getVariance();
	}

	/**This method returns a simulated value from the distribution*/
	public double simulate(){
		return location + scale * dist.simulate();
	}

	/**This method returns the cumulative distribution function*/
	public double getCDF(double x){
		if (scale > 0) return dist.getCDF((x - location) / scale);
		else return 1 - dist.getCDF((x - location) / scale);
	}

	/**This method returns the getQuantile function*/
	public double getQuantile(double p){
		if (scale > 0) return location + scale * dist.getQuantile(p);
		else return location + scale * dist.getQuantile(1 - p);
	}
}

