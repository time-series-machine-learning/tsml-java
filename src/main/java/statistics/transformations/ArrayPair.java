package statistics.transformations;
/*
 * Created on Jan 31, 2006
 *
 */

/**
 * @author ajb
 *
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
