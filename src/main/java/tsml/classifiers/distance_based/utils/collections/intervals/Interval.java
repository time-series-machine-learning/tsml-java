package tsml.classifiers.distance_based.utils.collections.intervals;

import tsml.classifiers.distance_based.utils.collections.params.ParamHandler;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;

/**
 * Purpose: represent an interval, i.e. some subsequence of indices. An interval therefore has a start and end point
 * (inclusively!). The start point may be beyond the end point to reverse directionality.
 *
 * Contributors: goastler
 */
public class Interval implements ParamHandler {
    private int start;
    private int end;
    public static final String START_FLAG = "s";
    public static final String END_FLAG = "e";

    public Interval() {
        this(-1, -1);
    }

    public Interval(final int start, final int end) {
        setEnd(end);
        setStart(start);
    }

    public int getEnd() {
        return end;
    }

    public void setEnd(final int end) {
        this.end = end;
    }

    public int getStart() {
        return start;
    }

    public void setStart(final int start) {
        this.start = start;
    }

    public int size() {
        return Math.abs(end - start) + 1;
    }

    /**
     * map interval index to instance index
     * @param index
     * @return
     */
    public int translate(int index) {
        index = adjustIndex(index);
        index += Math.min(start, end);
        return index;
    }

    public int adjustIndex(int index) {
        if(start > end) {
            index = size() - index - 1;
        }
        return index;
    }

    /**
     * map instance index to interval index
     * @param index
     * @return
     */
    public int inverseTranslate(int index) {
        index -= Math.min(start, end);
        index = adjustIndex(index);
        return index;
    }

    @Override public ParamSet getParams() {
        return ParamHandler.super.getParams().add(START_FLAG, start).add(END_FLAG, end);
    }

    @Override public void setParams(final ParamSet paramSet) throws Exception {
        ParamHandler.super.setParams(paramSet);
        ParamHandlerUtils.setParam(paramSet, START_FLAG, this::setStart, Integer.class);
        ParamHandlerUtils.setParam(paramSet, END_FLAG, this::setEnd, Integer.class);
    }
}
