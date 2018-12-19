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

/**This class defines a partition of an interval into subintervals of equal width.  These objects are used to define default domains.  A finite domain can be modeled by the values (midpoints) of the partition.  The boundary points are a + i * w for i = 0, ..., n, where n is the size of the partition, a is the lower bound and w the width. The values (midpoints) are a + (i + 1/2) * w, for i = 0, ..., n - 1.*/
public class Domain{
	//Variables
	private double lowerBound, upperBound, width, lowerValue, upperValue;
	private int size;

	/**This general constructor creates a new partition of a specified interval [a, b] into subintervals of width w*/
	public Domain(double a, double b, double w){
		if (w <= 0) w = 1;
		width = w;
		if (b < a + w) b = a + w;
		lowerBound = a; upperBound = b;
		lowerValue = lowerBound + 0.5 * width; upperValue = upperBound - 0.5 * width;
		size = (int)Math.rint((b - a) / w);
	}

	/**This special constructor creates a new partition of [0, b] into 10 equal subintervals*/
	public Domain(double b){
		this(0, b, 0.1 * b);
	}

	/**This default constructor creates a new partition of [0, 1] into 10 equal subintervals*/
	public Domain(){
		this(1);
	}

	/**This method returns the index of the interval containing a given value of x*/
	public int getIndex(double x){
		if (x < lowerBound) return -1;
		if (x > upperBound) return size;
		else return (int)Math.rint((x - lowerValue) / width);
	}

	/**This method returns the boundary point corresponding to a given index*/
	public double getBound(int i){
		return lowerBound + i * width;
	}

	/**This method return the midpoint of the interval corresponding to a given index*/
	public double getValue(int i){
		return lowerValue + i * width;
	}

	/**This method returns the lower bound*/
	public double getLowerBound(){
		return lowerBound;
	}

	/**This method returns the upper bound*/
	public double getUpperBound(){
		return upperBound;
	}

	/**This method returns the lower midpoint*/
	public double getLowerValue(){
		return lowerValue;
	}

	/**This method returns the upper midpoint*/
	public double getUpperValue(){
		return upperValue;
	}

	/**This method returns the width of the partition*/
	public double getWidth(){
		return width;
	}

	/**This method returns the size of the partition (the number of subintervals)*/
	public int getSize(){
		return size;
	}
}



