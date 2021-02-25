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

package tsml.data_containers;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

/**
 * Class to store a time series. The series can have different indices (time stamps) and store missing values (NaN).
 *
 * The model for the indexes is the first is always zero the other indexes are in units of md.increment
 * Hopefully most of this can be encapsulated, so if the data has equal increments then indices is null and the user
 *
 * @author Aaron Bostrom, 2020
 */
public class TimeSeries implements Iterable<Double> {

    public final static double DEFAULT_VALUE = Double.NaN;
    private final static List<Double> EMPTY_INDICES = Collections.emptyList();

    private List<Double> series;
    private List<Double> indices = EMPTY_INDICES;

    // just for internal use
    private TimeSeries() {}

    /**
     * Create a TimeSeries object from an array of time series data.
     *
     * @param data time series raw data
     */
    public TimeSeries(double[] data){
        series = new ArrayList<Double>();
        for(double value : data)
            series.add(value);
    }

    /**
     * Create a TimeSeries object from a list of time series data.
     *
     * @param data time series raw data
     */
    public TimeSeries(List<Double> data) {
        series = new ArrayList<>(data);
    }

    /**
     * Create a TimeSeries object from another TimeSeries object.
     *
     * @param other TimeSeries object
     */
    public TimeSeries(TimeSeries other) {
        this(other.series);
    }

    /**
     * Returns the length of the series.
     *
     * @return int length of series
     */
    public int getSeriesLength() {
        return series.size();
    }

    /**
     * Returns whether there is a valid value at the index passed.
     * i.e. if index is out of range or NaN, returns false.
     *
     * @param index to check
     * @return true if valid, false if not
     */
    public boolean hasValidValueAt(int index) {
        // test whether its out of range, or NaN
        return index < series.size() && Double.isFinite(series.get(index));
    }

    /**
     * Returns value at passed index.
     *
     * @param index to get value from
     * @return value at index
     */
    public double getValue(int index){
        return series.get(index);
    }

    /**
     * Returns a value at a specific index in the time series. This method conducts unboxing so use getValue if you care about performance.
     *
     * @param index to get value from
     * @return value at index
     */
    public Double get(int index) {
        return series.get(index);
    }

    /**
     * Returns value at index passed, or default value if no valid value at index.
     *
     * @param index to get value from
     * @return value at index, or default value if not valid at index
     */
    public double getOrDefault(int index) {
        return hasValidValueAt(index) ? getValue(index) : DEFAULT_VALUE;
    }

    /**
     * Returns a DoubleStream of values in series.
     *
     * @return stream of values in series
     */
    public DoubleStream streamValues() {
        return series.stream().mapToDouble(Double::doubleValue);
    }

    /**
     * Returns Stream of doubles for values in series.
     *
     * @return stream of doubles in series
     */
    public Stream<Double> stream() {
        return series.stream();
    }

    /**
     * Returns all values in series.
     *
     * @return values in series
     */
    public List<Double> getSeries() {
        return series;
    }

    /**
     * @return List<Double>
     */
    public List<Double> getIndices() {
        return indices;
    }

    /**
     * Returns the series, separated by commas.
     *
     * @return series, comma separated
     */
    @Override
    public String toString(){
        StringBuilder sb = new StringBuilder();

        for(double val : series) {
            sb.append(val).append(',');
        }

        return sb.substring(0, sb.length() - 1);
    }

    /**
     * Returns all values in the series.
     *
     * @return values in series
     */
	public double[] toValueArray() {
		return getSeries().stream().mapToDouble(Double::doubleValue).toArray();
    }

    /**
     * Returns a new TimeSeries object containing only values at indexes passed.
     *
     * @param indexesToKeep from series
     * @return a new TimeSeries object
     */
    public TimeSeries getVSlice(int[] indexesToKeep) {
        return new TimeSeries(getVSliceArray(indexesToKeep));
    }

    /**
     * Returns a new TimeSeries object containing only the value at the index
     * passed.
     *
     * @param indexToKeep from series
     * @return a new TimeSeries object
     */
    public TimeSeries getVSlice(int indexToKeep) {
	    return getVSlice(new int[] {indexToKeep});
    }

    /**
     * Returns a new TimeSeries object containing only values at indexes passed.
     *
     * @param indexesToKeep from series
     * @return a new TimeSeries object
     */
    public TimeSeries getVSlice(List<Integer> indexesToKeep) {
	    return getVSlice(indexesToKeep.stream().mapToInt(Integer::intValue).toArray());
    }

    /**
     * Returns a new TimeSeries object containing all values apart from index
     * passed.
     *
     * @param indexToRemove from series
     * @return a new TimeSeries object
     */
    public TimeSeries getVSliceComplement(int indexToRemove) {
	    return getVSliceComplement(new int[] {indexToRemove});
    }

    /**
     * Returns a new TimeSeries object containing all values apart from indexes
     * passed.
     *
     * @param indexesToRemove from series
     * @return a new TimeSeries object
     */
    public TimeSeries getVSliceComplement(int[] indexesToRemove) {
        return new TimeSeries(getVSliceComplementArray(indexesToRemove));
    }

    /**
     * Returns a new TimeSeries object containing all values apart from indexes
     * passed.
     *
     * @param indexesToRemove from series
     * @return a new TimeSeries object
     */
    public TimeSeries getVSliceComplement(List<Integer> indexesToRemove) {
        return getVSliceComplement(indexesToRemove.stream().mapToInt(Integer::intValue).toArray());
    }

    /**
     * Returns a list of series containing all values apart from indexes passed.
     *
     * This is useful if you want to delete a column/truncate the array, but
     * without modifying the original dataset.
     *
     * @param indexesToRemove from series
     * @return a list of new series
     */
    public List<Double> getVSliceComplementList(List<Integer> indexesToRemove){
        //if the current index isn't in the removal list, then copy across.
        List<Double> out = new ArrayList<>(this.getSeriesLength() - indexesToRemove.size());
        for(int i=0; i<this.getSeriesLength(); ++i){
            if(!indexesToRemove.contains(i))
                out.add(getOrDefault(i));
        }

        return out;
    }

    /**
     * Returns a list of series containing all values apart from indexes passed.
     *
     * @param indexesToRemove from series
     * @return a list of new series
     */
    public List<Double> getVSliceComplementList(int[] indexesToRemove) {
        return getVSliceComplementList(Arrays.stream(indexesToRemove).boxed().collect(Collectors.toList()));
    }

    /**
     * Returns a list of series containing all values apart from index passed.
     *
     * @param indexToRemove from series
     * @return a list of new series
     */
    public List<Double> getVSliceComponentList(int indexToRemove) {
        return getVSliceComplementList(new int[] {indexToRemove});
    }

    /**
     * Returns an array of series containing all values apart from indexes passed.
     *
     * @param indexesToRemove from series
     * @return an array of new series
     */
    public double[] getVSliceComplementArray(int[] indexesToRemove){
        return getVSliceComplementArray(Arrays.stream(indexesToRemove).boxed().collect(Collectors.toList()));
    }

    /**
     * Returns an array of series containing all values apart from indexes passed.
     *
     * @param indexesToRemove from series
     * @return an array of new series
     */
    public double[] getVSliceComplementArray(List<Integer> indexesToRemove){
        return getVSliceComplementList(indexesToRemove).stream().mapToDouble(Double::doubleValue).toArray();
    }

    /**
     * Returns an array of series containing all values apart from index passed.
     *
     * @param indexToRemove from series
     * @return an array of new series
     */
    public double[] getVSliceComplementArray(int indexToRemove) {
        return getVSliceComplementArray(new int[] {indexToRemove});
    }

    /**
     * Returns a list of series containing only values at indexes passed.
     *
     * @param indexesToKeep from series
     * @return a list of new series
     */
    public List<Double> getVSliceList(List<Integer> indexesToKeep){
        //if the current index isn't in the removal list, then copy across.
        List<Double> out = new ArrayList<>(indexesToKeep.size());
        for(int i=0; i<this.getSeriesLength(); ++i){
            if(indexesToKeep.contains(i))
                out.add(getOrDefault(i));
        }

        return out;
    }

    /**
     * Returns a list of series containing only values at indexes passed.
     *
     * @param indexesToKeep from series
     * @return a list of new series
     */
    public List<Double> getVSliceList(int[] indexesToKeep) {
        return getVSliceList(Arrays.stream(indexesToKeep).boxed().collect(Collectors.toList()));
    }

    /**
     * Returns a list of series containing only the value at index passed.
     *
     * @param indexToKeep from series
     * @return a list of new series
     */
    public List<Double> getVSliceList(int indexToKeep) {
        return getVSliceList(new int[] {indexToKeep});
    }

    /**
     * Returns an array of series containing only the value at index passed.
     *
     * @param indexToKeep from series
     * @return a list of new series
     */
    public double[] getVSliceArray(int indexToKeep) {
        return getVSliceArray(new int[] {indexToKeep});
    }

    /**
     * Returns an array of series containing only values at indexes passed.
     *
     * @param indexesToKeep from series
     * @return a list of new series
     */
    public double[] getVSliceArray(int[] indexesToKeep) {
        return getVSliceArray(Arrays.stream(indexesToKeep).boxed().collect(Collectors.toList()));
    }

    /**
     * Returns an array of series containing only values at indexes passed.
     *
     * @param indexesToKeep from series
     * @return a list of new series
     */
    public double[] getVSliceArray(List<Integer> indexesToKeep) {
        return getVSliceList(indexesToKeep).stream().mapToDouble(Double::doubleValue).toArray();
    }

    /**
     * Returns an iterator, iterating over the series.
     *
     * @return series iterator
     */
    @Override public Iterator<Double> iterator() {
        return series.iterator();
    }

    /**
     * Returns a list of the portion of the series between the specified start,
     * inclusive, and end, exclusive.
     *
     * @param startInclusive index to start from (inclusive)
     * @param endExclusive index to end from (exclusive)
     * @return Sliding window of series
     */
    public List<Double> getVSliceList(int startInclusive, int endExclusive) {
        return series.subList(startInclusive, endExclusive);
    }

    /**
     * Returns an array of the portion of the series between the specified start,
     * inclusive, and end, exclusive.
     *
     * @param startInclusive index to start from (inclusive)
     * @param endExclusive index to end from (exclusive)
     * @return Sliding window of series
     */
    public double[] getVSliceArray(int startInclusive, int endExclusive) {
        return getVSliceList(startInclusive, endExclusive).stream().mapToDouble(d -> d).toArray();
    }

    /**
     * Returns a new TimeSeries object containing a portion of the series between
     * the specified start, inclusive, and end, exclusive.
     *
     * @param startInclusive index to start from (inclusive)
     * @param endExclusive index to end from (exclusive)
     * @return Sliding window of series
     */
    public TimeSeries getVSlice(int startInclusive, int endExclusive) {
        final TimeSeries ts = new TimeSeries();
        ts.series = getVSliceList(startInclusive, endExclusive);
        return ts;
    }

    /**
     * Returns whether a TimeSeries object is equal to another based if the series
     * are exactly the same.
     *
     * @param other object
     * @return true if equal, false if not
     */
    @Override
    public boolean equals(final Object other) {
        if (!(other instanceof TimeSeries)) {
            return false;
        }
        final TimeSeries that = (TimeSeries) other;
        return Objects.equals(series, that.series);
    }

    /**
     * Returns an int of the hash code based on the series.
     *
     * @return hash code
     */
    @Override public int hashCode() {
        return Objects.hash(series);
    }

    /**
     * Example
     */
    public static void main(String[] args) {
        TimeSeries ts = new TimeSeries(new double[]{1,2,3,4});
    }
}
