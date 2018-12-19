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

/**This class models the discrete arcsine distribution that governs the last zero in
a symmetric random walk on an interval.*/
public class DiscreteArcsineDistribution extends Distribution{
	//Paramters
	private int parameter;

	/**This general constructor creates a new discrete arcsine distribution with a specified
	number of steps.*/
	public DiscreteArcsineDistribution(int n){
		setParameter(n);
	}

	/**This default constructor creates a new discrete arcsine distribution with 10 steps.*/
	public DiscreteArcsineDistribution(){
		this(10);
	}

	/**This method sets the parameter, the number of steps.*/
	public void setParameter(int n){
		parameter = n;
		setParameters(0, parameter, 2, DISCRETE);
	}

	/**This method computes the density function.*/
	public double getDensity(double x){
		int k = (int)x;
		return comb(k, k / 2) * comb(parameter - k, (parameter - k) / 2) / Math.pow(2, parameter);
	}

	/**This method computes the maximum value of the density function.*/
	public double getMaxDensity(){
		return getDensity(0);
	}

	/**This method gets the parameter, the number of steps.*/
	public int getParameter(){
		return parameter;
	}

	/**This method simulates a value from the distribution, by simulating a random walk on the
	interval.*/
	public double simulate(){
		int step, lastZero = 0, position = 0;
		for (int i = 1; i <= parameter; i++){
			if (Math.random() < 0.5) step = 1;
			else step = -1;
			position = position + step;
			if (position == 0) lastZero = i;
		}
		return lastZero;
	}
}

