package transformations;
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
