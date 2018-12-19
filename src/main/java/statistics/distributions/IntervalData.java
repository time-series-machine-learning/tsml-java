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

/**This class defines a simple implementation of an interval data distribution. The data distribution is based on a specified domain (that is, a partition of an interval). When values are added, frequency counts for the subintervals are computed and various statistic updated.*/
public class IntervalData{
	//Variables
	private int size, maxFreq;
	private double value, minValue, maxValue, mean, meanSquare, mode;
	private int[] freq;
	//Objects
	private Domain domain;
	private String name;

	/**This general constructor creates a new data distribution with a specified domain
	and a specified name*/
	public IntervalData(Domain d, String n){
		name = n;
		setDomain(d);
	}

	/**This general constructor creates a new data distribution with a specified domain and a
	specified name.*/
	public IntervalData(double a, double b, double w, String n){
		this(new Domain(a, b, w), n);
	}

	/**This special constructor creates a new data distribution with a specified domain and the
	default name "X".*/
	public IntervalData(Domain d){
		this(d, "X");
	}

	/**This spcial constructor creates a new data distribution with a specified domain and
	the name "X"*/
	public IntervalData(double a, double b, double w){
		this(a, b, w, "X");
	}

	/**This default constructor creates a new data distribution on the interval [0, 1] with
	subintervals of length 0.1, and the default name "X".*/
	public IntervalData(){
		this(0, 1, 0.1);
	}

	/**This method sets the domain of the data set.*/
	public void setDomain(Domain d){
		domain = d;
		reset();
	}

	/**This method returns the domain.*/
	public Domain getDomain(){
		return domain;
	}

	/**This method sets the name of the data set.*/
	public void setName(String n){
		name = n;
	}

	/**This method gets the name of the data set.*/
	public String getName(){
		return name;
	}

	/**This method resets the data set*/
	public void reset(){
		freq = new int[domain.getSize()];
		size = 0;
		minValue = domain.getUpperBound();
		maxValue = domain.getLowerBound();
		maxFreq = 0;
	}

	/**This method adds a new number to the data set and re-compute the mean, mean square,
	minimum and maximum values, the frequency distribution, and the mode*/
	public void setValue(double x){
		value = x;
		//Update the size of the data set:
		size++;
		//Re-compute mean and mean square
		mean = ((double)(size - 1) / size) * mean + value / size;
		meanSquare = ((double)(size - 1) / size) * meanSquare + value * value / size;
		//Recompute minimum and maximum values
		if (value < minValue) minValue = value;
		if (value > maxValue) maxValue = value;
		//Update frequency distribution
		int i = domain.getIndex(x);
		if (i >= 0 & i < domain.getSize()){
			freq[i]++;
			//Re-compute mode
			if (freq[i] > maxFreq){
				maxFreq = freq[i];
				mode = domain.getValue(i);
			}
			else if (freq[i] == maxFreq) mode = Double.NaN; //There are two or more modes
		}
	}

	/**This method returns the current value of the data set*/
	public double getValue(){
		return value;
	}

	/**This method returns the domain value (midpoint) closest to given value of x*/
	public double getDomainValue(double x){
		return domain.getValue(domain.getIndex(x));
	}

	/**This method returns the frequency of the class containing a given value
	of x.*/
	public int getFreq(double x){
		int i = domain.getIndex(x);
		if (i < 0 | i >= domain.getSize()) return 0;
		else return freq[i];
	}

	/**This method returns the relative frequency of the class containing
	a given value.*/
	public double getRelFreq(double x){
		if (size > 0) return (double)(getFreq(x)) / size;
		else return 0;
	}

	/**This method returns the getDensity for a given value*/
	public double getDensity(double x){
		return getRelFreq(x) / domain.getWidth();
	}

	/**This method returns the mean of the data set.*/
	public double getMean(){
		return mean;
	}

	/**This method returns the mean of the frequency distribution.  The interval
	mean is an approximation to the true mean of the data set.*/
	public double getIntervalMean(){
		double sum = 0;
		for (int i = 0; i < domain.getSize(); i++) sum = sum + domain.getValue(i) * freq[i];
		return sum / size;
	}

	/**This method returns the population variance*/
	public double getVarianceP(){
		double var = meanSquare - mean * mean;
		if (var < 0) var = 0;
		return var;
	}

	/**This method returns the population standard deviation.*/
	public double getSDP(){
		return Math.sqrt(getVarianceP());
	}

	/**This method returns the sample variance.*/
	public double getVariance(){
		return ((double)size / (size - 1)) * getVarianceP();
	}

	/**This method returns the sample standard deviation.*/
	public double getSD(){
		return Math.sqrt(getVariance());
	}

	/**This method returns the interval variance.*/
	public double getIntervalVariance(){
		double m = getIntervalMean(), sum = 0, x;
		for (int i = 0; i < domain.getSize(); i++){
			x = domain.getValue(i);
			sum = sum + (x - m) * (x - m) * freq[i];
		}
		return sum / size;
	}

	/**This method returns the interval standard deviation.*/
	public double getIntervalSD(){
		return Math.sqrt(getIntervalVariance());
	}

	/**This method returns the minimum value of the data set*/
	public double getMinValue(){
		return minValue;
	}

	/**This method returns the maximum value of the data set*/
	public double getMaxValue(){
		return maxValue;
	}

	/**This method computes the median of the values in the data set between two specified values*/
	public double getMedian(double a, double b){
		int sumFreq = 0, numValues = 0, lRank, uRank;
		double lValue = a - 1, uValue = b + 1, w = domain.getWidth();
		//Compute sum of frequencies between a and b
		for (double x = a; x <= b + 0.5 * w; x = x + w) numValues = numValues + getFreq(x);
		//Determine parity and ranks
		if (2 * (numValues / 2) == numValues) {
			lRank = numValues / 2;
			uRank = lRank + 1;
		}
		else {
			lRank = (numValues + 1) / 2;
			uRank = lRank;
		}
		//Determine values
		for (double x = a; x <= b + 0.5 * w; x = x + w) {
			sumFreq = sumFreq + getFreq(x);
			if ((lValue == a - 1) & (sumFreq >= lRank)) lValue = x;
			if ((uValue == b + 1) & (sumFreq >= uRank)) uValue = x;
		}
		//Return average of upper and lower values
		return (uValue + lValue) / 2;
	}

	/**This method computes the median of the entire data set*/
	public double getMedian(){
		return getMedian(domain.getLowerValue(), domain.getUpperValue());
	}

	/**This method returns the quartiles of the data set.*/
	public double getQuartile(int i){
		if (i < 1) i = 1; else if (i > 3) i = 3;
		if (i == 1) return getMedian(domain.getLowerValue(), getMedian());
		else if (i == 2) return getMedian();
		else return getMedian(getMedian(), domain.getUpperValue());
	}

	/**This method computes the mean absoulte deviation*/
	public double getMAD(){
		double mad = 0, x;
		double m = getMedian();
		for (int i = 0; i < domain.getSize(); i++){
			x = domain.getValue(i);
			mad = mad + getRelFreq(x) * Math.abs(x - m);
		}
		return mad;
	}

	/**This method returns the number of pointCount in the data set*/
	public int getSize(){
		return size;
	}

	/**This method returns the maximum frequency*/
	public int getMaxFreq(){
		return maxFreq;
	}

	/**This method returns the maximum relative frequency.*/
	public double getMaxRelFreq(){
		if (size > 0) return (double)maxFreq / size;
		else return 0;
	}

	/**This method returns the maximum getDensity.*/
	public double getMaxDensity(){
		return getMaxRelFreq() / domain.getWidth();
	}

	/**This method returns the mode of the distribution. The mode may not exist*/
	public double getMode(){
		return mode;
	}
}

