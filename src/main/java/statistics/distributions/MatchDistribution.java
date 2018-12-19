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

/**The distribution of the number of matches in a random permutation*/
public class MatchDistribution extends Distribution{
	int parameter;
	int[] b;

	/**This general constructor creates a new matching distribution with a
	specified parameter*/
	public MatchDistribution(int n){
		setParameter(n);
	}

	/**this default constructor creates a new mathcing distribuiton with
	parameter 5*/
	public MatchDistribution(){
		this(5);
	}

	/**This method sets the parameter of the distribution (the size of the
	random permutation*/
	public void setParameter(int n){
		if (n < 1) n = 1;
		parameter = n;
		super.setParameters(0, parameter, 1, DISCRETE);
		b = new int[n];
	}

	/**This method computes the getDensity function*/
	public double getDensity(double x){
		int k = (int)Math.rint(x);
		double sum = 0;
		int sign = -1;
		for (int j = 0; j <= parameter - k; j++){
			sign = -sign;
			sum = sum + sign / factorial(j);
		}
		return sum / factorial(k);
	}

	/**This method gives the maximum value of the getDensity function*/
	public double getMaxDensity(){
		if (parameter == 2) return getDensity(0);
		else return getDensity(1);
	}

	/**This method returns the mean*/
	public double getMean(){
		return 1;
	}

	/**This method returns the variance*/
	public double getVariance(){
		return 1;
	}

	/**This method gets the parameter*/
	public int getParameter(){
		return parameter;
	}

	/**This method simulates a value from the distribution, by generating
	a random permutation and computing the number of matches*/
	public double simulate(){
		int j, k, u;
		double matches = 0;
		for (int i = 0; i < parameter; i++) b[i] = i + 1;
		for (int i = 0; i < parameter; i++){
			j = parameter - i;
			u = (int)(j * Math.random());
			if (b[u] == i + 1) matches = matches + 1;
			k = b[j - 1];
			b[j - 1] = b[u];
			b[u] = k;
		}
		return matches;
	}
}

