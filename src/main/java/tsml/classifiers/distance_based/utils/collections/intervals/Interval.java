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
 
package tsml.classifiers.distance_based.utils.collections.intervals;

import org.junit.Assert;
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
    private int length;
    public static final String START_FLAG = "s";
    public static final String LENGTH_FLAG = "l";

    public Interval() {
        this(0, 0);
    }

    public Interval(final int start, final int length) {
        setLength(length);
        setStart(start);
    }

    public boolean contains(int index) {
        return start <= index && index < start + length;
    }

    public int getLength() {
        return length;
    }

    public void setLength(final int length) {
        Assert.assertTrue(length >= 0);
        this.length = length;
    }

    public int getStart() {
        return start;
    }

    public void setStart(final int start) {
        Assert.assertTrue(start >= 0);
        this.start = start;
    }

    public int size() {
        return length;
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
            if(index > length - 1) {
                throw new ArrayIndexOutOfBoundsException(index);
            }
            if(index < 0) {
                throw new ArrayIndexOutOfBoundsException(index);
            }
        }
        return index + start;
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
            if(index > start + length - 1) {
                throw new ArrayIndexOutOfBoundsException(index);
            }
            if(index < start) {
                throw new ArrayIndexOutOfBoundsException(index);
            }
        }
        return index - start;
    }

    @Override public ParamSet getParams() {
        return ParamHandler.super.getParams().add(START_FLAG, start).add(LENGTH_FLAG, length);
    }

    @Override public void setParams(final ParamSet paramSet) throws Exception {
        ParamHandler.super.setParams(paramSet);
        ParamHandlerUtils.setParam(paramSet, START_FLAG, this::setStart, Integer.class);
        ParamHandlerUtils.setParam(paramSet, LENGTH_FLAG, this::setLength, Integer.class);
    }

    @Override public String toString() {
        return "Interval{" +
               "start=" + start +
               ", length=" + length +
               '}';
    }

    public int getEnd() {
        return start + length - 1;
    }
}
