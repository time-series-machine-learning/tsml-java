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

/**This class models the crcle distribution with parameter a.  This is the distribution of X
and Y when (X, Y) has the uniform distribution on a circular region with a specified radius.*/
public class CircleDistribution extends Distribution{
	private double radius;

	/**This general constructor creates a new circle distribution with a specified radius.*/
	public CircleDistribution(double r){
		setRadius(r);
	}

	/**This special constructor creates a new circle distribution with radius 1*/
	public CircleDistribution(){
		this(1);
	}

	/**This method sets the radius parameter*/
	public void setRadius(double r){
		if (r <= 0) r =1;
		radius = r;
		super.setParameters(-radius, radius, 0.02 * radius, CONTINUOUS);
	}

	/**This method computes the getDensity function.*/
	public double getDensity(double x){
		if (-radius <= x & x <= radius)
			return 2 * Math.sqrt(radius * radius - x * x) / (Math.PI * radius * radius);
		else return 0;
	}

	/**This method computes the maximum value of the getDensity function.*/
	public double getMaxDensity(){
		return getDensity(0);
	}

	/**This method computes the mean*/
	public double getMean(){
		return 0;
	}

	/**This method computes the variance*/
	public double getVariance(){
		return radius * radius / 4;
	}

	/**This method computes the median.*/
	public double getMedian(){
		return 0;
	}

	/**This method returns the radius parameter.*/
	public double getRadius(){
		return radius;
	}

	/**This method simulates a value from the distribution.*/
	public double simulate(){
		double u = radius * Math.random();
		double v = radius * Math.random();
		double r = Math.max(u, v);
		double theta = 2 * Math.PI * Math.random();
		return r * Math.cos(theta);
	}

	/**This method compute the cumulative distribution function.*/
	public double getCDF(double x){
		return 0.5 + Math.asin(x / radius) / Math.PI
			+ x * Math.sqrt(1 - x * x / (radius * radius)) / (Math.PI * radius);
	}
}

