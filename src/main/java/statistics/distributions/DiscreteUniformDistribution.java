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

/**The discrete uniform distribution on a finite set*/
public class DiscreteUniformDistribution extends Distribution{
	double values;

	public DiscreteUniformDistribution(double a, double b, double w){
		setParameters(a, b, w);
	}

	public DiscreteUniformDistribution(){
		this(1, 6, 1);
	}

	public void setParameters(double a, double b, double w){
		super.setParameters(a, b, w, DISCRETE);
	}

	public double getDensity(double x){
		if (getDomain().getLowerValue() <= x & x <= getDomain().getUpperValue()) return 1.0 / getDomain().getSize();
		else return 0;
	}

	public double getMaxDensity(){
		return 1.0 / getDomain().getSize();
	}

	public double simulate(){
		return getDomain().getLowerValue() + Math.random() * getDomain().getSize() * getDomain().getWidth();
	}
}

