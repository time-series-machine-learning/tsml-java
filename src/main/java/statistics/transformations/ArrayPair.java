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
public class ArrayPair implements Comparable {
	public double predicted;
	public double residual;
	public int compareTo(Object c) {
		if(this.predicted>((ArrayPair)c).predicted)
			return 1;
		else if(this.predicted<((ArrayPair)c).predicted)
			return -1;
		return 0;
	}

	public static void main(String[] args) {
	}
}
