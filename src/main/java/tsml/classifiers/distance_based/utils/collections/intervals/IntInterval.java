package tsml.classifiers.distance_based.utils.collections.intervals;

/**
 * Purpose: represent an interval, i.e. some subsequence of indices. An interval therefore has a start and end point
 * (inclusively!). The start point may be beyond the end point to reverse directionality.
 *
 * Contributors: goastler
 */
public class IntInterval extends BaseInterval<Integer> {

    public IntInterval() {
        this(0, 0);
    }

    public IntInterval(final int start, final int end) {
        super(start, end);
    }

    public boolean contains(Integer index) { 
        return getStart() <= index && index <= getEnd();
    }

    public int translate(int index) {
        return translate(index, true);
    }

    /**
     * map interval index to instance index
     * @param index
     * @return
     */
    public int translate(int index, boolean check) {
        if(check) {
            if(index > size() - 1) {
                throw new ArrayIndexOutOfBoundsException(index);
            }
            if(index < 0) {
                throw new ArrayIndexOutOfBoundsException(index);
            }
        }
        return index + getStart();
    }

    /**
     * map instance index to interval index
     * @param index
     * @return
     */
    public int inverseTranslate(int index) {
        return inverseTranslate(index, true);
    }

    public int inverseTranslate(int index, boolean check) {
        if(check) {
            if(index > getEnd()) {
                throw new ArrayIndexOutOfBoundsException(index);
            }
            if(index < getStart()) {
                throw new ArrayIndexOutOfBoundsException(index);
            }
        }
        return index - getStart();
    }

    @Override public Integer size() {
        return getEnd() - getStart() + 1;
    }
}
