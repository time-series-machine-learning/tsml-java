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

/**A basic discrete distribution on a finite set of points, with specified
probabilities*/
public class FiniteDistribution extends Distribution{
	private int n;
	private double[] prob;

	/**Constructs a new finite distribution on a finite set of points with
	a specified array of probabilities*/
	public FiniteDistribution(double a, double b, double w, double[] p){
		setParameters(a, b, w, p);
	}

	/**Constructs the uniform distribuiton on the finite set of points*/
	public FiniteDistribution(double a, double b, double w){
		super.setParameters(a, b, w, DISCRETE);
		n = getDomain().getSize();
		prob = new double[n];
		for (int i = 0; i < n; i++) prob[i] = 1.0 / n;
	}

	/**This special constructor creates a new uniform distribution on {1, 2, ..., 10}.*/
	public FiniteDistribution(){
		this(1, 10, 1);
	}

	/**This method sets the parameters: the domain and the probabilities.*/
	public void setParameters(double a, double b, double w, double[] p){
		super.setParameters(a, b, w, DISCRETE);
		n = getDomain().getSize();
		prob = new double[n];
		if (p.length != n) p = new double[n];
		double sum = 0;
		for (int i = 0; i < n; i++){
			if (p[i] < 0) p[i] = 0;
			sum = sum + p[i];
		}
		if (sum == 0) for (int i = 0; i < n; i++) prob[i] = 1.0 / n;
		else for (int i = 0; i < n; i++) prob[i] = p[i] / sum;
	}

	/**Density function*/
	public double getDensity(double x){
		int j = getDomain().getIndex(x);
		if (0 <= j & j < n) return prob[j];
		else return 0;
	}

	/**Set the probabilities*/
	public void setProbabilities(double[] p){
		if (p.length != n) p = new double[n];
		double sum = 0;
		for (int i = 0; i < n; i++){
			if (p[i] < 0) p[i] = 0;
			sum = sum + p[i];
		}
		if (sum == 0) for (int i = 0; i < n; i++) prob[i] = 1.0 / n;
		else for (int i = 0; i < n; i++) prob[i] = p[i] / sum;
	}

	/**This method gets the probability for a specified index*/
	public double getProbability(int i){
		if (i < 0) i = 0; else if (i >= n) i = n - 1;
		return prob[i];
	}

	/**This method gets the probability vector.*/
	public double[] getProbabilities(){
		return prob;
	}
}



