/*
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */
 

package statistics.transformations;
/*
 * Created on Jan 31, 2006
 *
 * TODO To change the template for this generated file go to
 * Window - Preferences - Java - Code Style - Code Templates
 */

/**
 * @author ajb
 *
 * TODO To change the template for this generated type comment go to
 * Window - Preferences - Java - Code Style - Code Templates
 */
public class MatrixSort implements Comparable {
	public double[] x;
	public double y;
	public int pos=0;
	public MatrixSort(double[] X, double Y, int p)
	{
		x=X;
		y=Y;
		pos=p;
	}
	public int compareTo(Object c) {
		
		if(this.x[pos]>((MatrixSort)c).x[pos])
			return 1;
		else if(this.x[pos]<((MatrixSort)c).x[pos])
			return -1;
		return 0;
	}

	public static void main(String[] args) {
	}
}
