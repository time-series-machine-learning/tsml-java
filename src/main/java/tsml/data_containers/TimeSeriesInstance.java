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
import java.util.stream.Stream;

/**
 * Data structure able to store a time series instance. it can be standard
 * (univariate, no missing, equally sampled series) or complex (multivariate,
 * unequal length, unequally spaced, univariate or multivariate time series).
 *
 * Should Instances be immutable after creation? Meta data is calculated on
 * creation, mutability can break this
 *
 * @author Aaron Bostrom, 2020
 */
public class TimeSeriesInstance implements Iterable<TimeSeries> {

    /* Meta Information */
    private boolean isMultivariate;
    private boolean isEquallySpaced; // todo compute whether timestamps are equally spaced
    private boolean hasMissing;
    private boolean isEqualLength;

    private int minLength;
    private int maxLength;

    /**
     * Returns whether data is multivariate.
     *
     * @return true if multivariate, false if not
     */
    public boolean isMultivariate() {
        return isMultivariate;
    }

    /**
     * Returns whether data is equally spaced.
     *
     * @return true if equally spaced, false if not
     */
    public boolean isEquallySpaced() {
        return isEquallySpaced;
    }

    /**
     * Returns whether data has missing values in.
     *
     * @return true if missing values, false if not
     */
    public boolean hasMissing() {
        return hasMissing;
    }

    /**
     * Returns whether data is equal length.
     *
     * @return true if equal length, false if not
     */
    public boolean isEqualLength() {
        return isEqualLength;
    }

    /**
     * Returns the minimum length of the data.
     *
     * @return minimum length
     */
    public int getMinLength() {
        return minLength;
    }

    /**
     * Returns the maximum length of the data.
     *
     * @return maximum length
     */
    public int getMaxLength() {
        return maxLength;
    }

    /* End Meta Information */


    /* Data */
    private List<TimeSeries> seriesDimensions;
    private int labelIndex = -1;
    private double targetValue = Double.NaN;

    /**
     * Create a TimeSeriesInstance object from a target value and a list of time
     * series data. For regression.
     *
     * @param targetValue value
     * @param series      raw data
     */
    public TimeSeriesInstance(double targetValue, List<? extends TimeSeries> series) {
        this.seriesDimensions = new ArrayList<>(series);
        this.targetValue = targetValue;

        dataChecks();
    }

    /**
     * Create a TimeSeriesInstance object from a label index and a list of time
     * series data. For classification.
     *
     * @param labelIndex of class label
     * @param series     raw data
     */
    public TimeSeriesInstance(int labelIndex, List<? extends TimeSeries> series) {
        this.seriesDimensions = new ArrayList<>(series);
        this.labelIndex = labelIndex;

        dataChecks();
    }

    /**
     * Construct a labelled instance from raw data.
     *
     * @param series raw data
     * @param label  target
     */
    public TimeSeriesInstance(List<? extends List<Double>> series, int label) {
        this(series, Double.NaN);

        targetValue = labelIndex = label;

        dataChecks();
    }

    public TimeSeriesInstance(List<? extends List<Double>> series, double targetValue) {
        // process the input list to produce TimeSeries Objects.
        // this allows us to pad if need be, or if we want to squarify the data etc.
        seriesDimensions = new ArrayList<TimeSeries>();

        for (List<Double> ts : series) {
            seriesDimensions.add(new TimeSeries(ts));
        }

        this.targetValue = targetValue;

        dataChecks();
    }

    /**
     * Construct an regressed instance from raw data.
     *
     * @param data        series
     * @param targetValue
     */
    public TimeSeriesInstance(double[][] data, double targetValue) {
        seriesDimensions = new ArrayList<TimeSeries>();

        for (double[] in : data) {
            seriesDimensions.add(new TimeSeries(in));
        }

        this.targetValue = targetValue;

        dataChecks();
    }

    /**
     * Construct an labelled instance from raw data.
     *
     * @param data        series
     * @param labelIndex
     * @param classLabels
     */
    public TimeSeriesInstance(double[][] data, int labelIndex, String[] classLabels) {
        seriesDimensions = new ArrayList<TimeSeries>();

        for (double[] in : data) {
            seriesDimensions.add(new TimeSeries(in));
        }

        targetValue = this.labelIndex = labelIndex;

        dataChecks();
    }

    /**
     * Construct an regressed instance from raw data.
     *
     * @param data
     * @param labelIndex
     */
    public TimeSeriesInstance(double[][] data, int labelIndex) {
        seriesDimensions = new ArrayList<TimeSeries>();

        for (double[] in : data) {
            seriesDimensions.add(new TimeSeries(in));
        }
        this.labelIndex = labelIndex;

        dataChecks();
    }

    /**
     * Returns a discretised label index.
     *
     * @param labelIndex to discretise
     * @return discretised label index
     */
    public static int discretiseLabelIndex(double labelIndex) {
        final int i;
        if (Double.isNaN(labelIndex)) {
            i = -1;
        }
        else {
            i = (int) labelIndex;
            /*
             Check the given double is an integer, i.e. 3.0 == 3.
             Protects against abuse through implicit label indexing integer casting, i.e. 3.3 --> 3.
             The user should do this themselves, otherwise it's safest to assume
             a non-integer value (e.g. 7.4) is an error and raise exception.
             */

            if (labelIndex != i) {
                throw new IllegalArgumentException("cannot discretise " + labelIndex + " to an int: " + i);
            }
        }
        return i;
    }

    /**
     * Construct a labelled instance from raw data with label in double form
     * (but should be an integer value).
     *
     * @param data
     * @param labelIndex
     * @param classLabels
     */
    public TimeSeriesInstance(double[][] data, double labelIndex, String[] classLabels) {
        this(data, discretiseLabelIndex(labelIndex), classLabels);
    }

    /**
     * Construct an instance from raw data. Copies over regression target /
     * labelling variables. This is only intended for internal use in avoiding
     * copying the data again after a vslice / hslice.
     *
     * @param data  series
     * @param other TimeSeriesInstance
     */
    private TimeSeriesInstance(double[][] data, TimeSeriesInstance other) {
        this(data, Double.NaN);
        labelIndex = other.labelIndex;
        targetValue = other.targetValue;

        dataChecks();
    }

    /**
     * Create a TimeSeriesInstance object from raw data.
     *
     * @param data series
     */
    public TimeSeriesInstance(double[][] data) {
        this(data, Double.NaN);
    }

    /**
     * Create a TimeSeriesInstance object from raw data.
     *
     * @param data series
     */
    public TimeSeriesInstance(List<? extends List<Double>> data) {
        this(data, Double.NaN);
    }

    private TimeSeriesInstance() {
    }

    /**
     * Returns a deep copy of a TimeSeriesInstance object.
     *
     * @return deep copy
     */
    private TimeSeriesInstance copy() {
        final TimeSeriesInstance inst = new TimeSeriesInstance();
        inst.labelIndex = labelIndex;
        inst.seriesDimensions = seriesDimensions;
        inst.targetValue = targetValue;

        inst.dataChecks();

        return inst;
    }

    public TimeSeriesInstance(double targetValue, TimeSeries[] data) {
        this(targetValue, Arrays.asList(data));
    }

    public TimeSeriesInstance(int labelIndex, TimeSeries[] data) {
        this(labelIndex, Arrays.asList(data));
    }

    /**
     * Performs data checks to calculate what types of data are inside.
     */
    private void dataChecks() {

        if (seriesDimensions == null) {
            throw new NullPointerException("no series dimensions");
        }

        calculateIfMultivariate();
        calculateLengthBounds();
        calculateIfMissing();
    }

    /**
     * Calculates whether the data is multivariate.
     */
    private void calculateIfMultivariate() {
        isMultivariate = seriesDimensions.size() > 1;
    }

    /**
     * Calculates the length bounds of the data.
     * (Minimum length, maximum length and equal length)
     */
    private void calculateLengthBounds() {
        minLength = seriesDimensions.stream().mapToInt(TimeSeries::getSeriesLength).min().getAsInt();
        maxLength = seriesDimensions.stream().mapToInt(TimeSeries::getSeriesLength).max().getAsInt();
        isEqualLength = minLength == maxLength;
    }

    /**
     * Calculates whether the data has missing values.
     */
    private void calculateIfMissing() {
        // if any of the series have a NaN value, across all dimensions then this is true.
        hasMissing = seriesDimensions.stream().anyMatch(e -> e.streamValues().anyMatch(Double::isNaN));
    }

    /**
     * Returns how many dimensions there are in the series.
     *
     * @return number of dimensions
     */
    public int getNumDimensions() {
        return seriesDimensions.size();
    }

    /**
     * Returns the label index.
     *
     * @return label index
     */
    public int getLabelIndex() {
        return labelIndex;
    }

    /**
     * Returns a list of values from each dimension in the series at the index.
     *
     * @param dimensionIndex to get values from
     * @return a list of values
     */
    public List<Double> getVSliceList(int dimensionIndex) {
        List<Double> out = new ArrayList<>(getNumDimensions());
        for (TimeSeries ts : seriesDimensions) {
            out.add(ts.getValue(dimensionIndex));
        }

        return out;
    }

    /**
     * Returns an array of values from the dimension in the series at the index.
     *
     * @param dimensionIndex to get values from
     * @return an array of values
     */
    public double[] getVSliceArray(int dimensionIndex) {
        double[] out = new double[getNumDimensions()];
        int i = 0;
        for (TimeSeries ts : seriesDimensions) {
            out[i++] = ts.getValue(dimensionIndex);
        }

        return out;
    }

    /**
     * Returns a 2d list: a list for each dimension; of values at the indexes.
     *
     * @param indexesToKeep to get values from
     * @return a 2d list of values
     */
    public List<List<Double>> getVSliceList(int[] indexesToKeep) {
        return getVSliceList(Arrays.stream(indexesToKeep).boxed().collect(Collectors.toList()));
    }

    /**
     * Returns a 2d list: a list for each dimension; of values at the indexes.
     *
     * @param indexesToKeep to get values from
     * @return a 2d list of values
     */
    public List<List<Double>> getVSliceList(List<Integer> indexesToKeep) {
        List<List<Double>> out = new ArrayList<>(getNumDimensions());
        for (TimeSeries ts : seriesDimensions) {
            out.add(ts.getVSliceList(indexesToKeep));
        }

        return out;
    }

    /**
     * Returns a 2d array: an array for each dimension; of values at the indexes.
     *
     * @param indexesToKeep to get values from
     * @return a 2d array of values
     */
    public double[][] getVSliceArray(int[] indexesToKeep) {
        return getVSliceArray(Arrays.stream(indexesToKeep).boxed().collect(Collectors.toList()));
    }

    /**
     * Returns a 2d array: an array for each dimension; of values at the indexes.
     *
     * @param indexesToKeep to get values from
     * @return a 2d array of values
     */
    public double[][] getVSliceArray(List<Integer> indexesToKeep) {
        double[][] out = new double[getNumDimensions()][];
        int i = 0;
        for (TimeSeries ts : seriesDimensions) {
            out[i++] = ts.getVSliceArray(indexesToKeep);
        }

        return out;
    }

    /**
     * Returns a TimeSeriesInstance containing values at the indexes from each
     * dimension.
     *
     * @param indexesToKeep to get values from
     * @return a new TimeSeriesInstance
     */
    public TimeSeriesInstance getVSlice(List<Integer> indexesToKeep) {
        return new TimeSeriesInstance(getVSliceArray(indexesToKeep), this);
    }

    /**
     * Returns a TimeSeriesInstance containing values at the indexes from each
     * dimension.
     *
     * @param indexesToKeep to get values from
     * @return a new TimeSeriesInstance
     */
    public TimeSeriesInstance getVSlice(int[] indexesToKeep) {
        return getVSlice(Arrays.stream(indexesToKeep).boxed().collect(Collectors.toList()));
    }

    /**
     * Returns a TimeSeriesInstance containing values at the index from each
     * dimension.
     *
     * @param index to get values from
     * @return a new TimeSeriesInstance
     */
    public TimeSeriesInstance getVSlice(int index) {
        return getVSlice(new int[]{index});
    }

    /**
     * Returns the series at the dimension passed.
     *
     * @param dim to get
     * @return list of series
     */
    public List<Double> getHSliceList(int dim) {
        return seriesDimensions.get(dim).getSeries();
    }


    /**
     * Returns the series at the dimension passed.
     *
     * @param dim to get
     * @return array of series
     */
    public double[] getHSliceArray(int dim) {
        return seriesDimensions.get(dim).toValueArray();
    }


    /**
     * Returns a 2d list: a list for each dimension index passed; containing the
     * series.
     *
     * TODO: not a clone. may need to be careful...
     *
     * @param dimensionsToKeep indexes
     * @return 2d list of series
     */
    public List<List<Double>> getHSliceList(int[] dimensionsToKeep) {
        return getHSliceList(Arrays.stream(dimensionsToKeep).boxed().collect(Collectors.toList()));
    }

    /**
     * Returns a 2d list: a list for each dimension index passed; containing the
     * series.
     *
     * TODO: not a clone. may need to be careful...
     *
     * @param dimensionsToKeep indexes
     * @return 2d list of series
     */
    public List<List<Double>> getHSliceList(List<Integer> dimensionsToKeep) {
        List<List<Double>> out = new ArrayList<>(dimensionsToKeep.size());
        for (Integer dim : dimensionsToKeep)
            out.add(seriesDimensions.get(dim).getSeries());

        return out;
    }

    /**
     * Returns a 2d array: an array for each dimension index passed; containing the
     * series.
     *
     * @param dimensionsToKeep indexes
     * @return 2d array of series
     */
    public double[][] getHSliceArray(int[] dimensionsToKeep) {
        return getHSliceArray(Arrays.stream(dimensionsToKeep).boxed().collect(Collectors.toList()));
    }

    /**
     * Returns a 2d array: an array for each dimension index passed; containing the
     * series.
     *
     * @param dimensionsToKeep indexes
     * @return 2d array of series
     */
    public double[][] getHSliceArray(List<Integer> dimensionsToKeep) {
        double[][] out = new double[dimensionsToKeep.size()][];
        int i = 0;
        for (Integer dim : dimensionsToKeep) {
            out[i++] = seriesDimensions.get(dim).toValueArray();
        }

        return out;
    }

    /**
     * Returns a TimeSeriesInstance containing each dimension of series from
     * indexes passed.
     *
     * @param dimensionsToKeep indexes
     * @return a new TimeSeriesInstance
     */
    public TimeSeriesInstance getHSlice(List<Integer> dimensionsToKeep) {
        return new TimeSeriesInstance(getHSliceArray(dimensionsToKeep), this);
    }

    /**
     * Returns a TimeSeriesInstance containing each dimension of series from
     * indexes passed.
     *
     * @param dimensionsToKeep indexes
     * @return a new TimeSeriesInstance
     */
    public TimeSeriesInstance getHSlice(int[] dimensionsToKeep) {
        return getHSlice(Arrays.stream(dimensionsToKeep).boxed().collect(Collectors.toList()));
    }

    /**
     * Returns a TimeSeriesInstance containing the dimension of series from
     * index passed.
     *
     * @param dimensionToKeep indexes
     * @return a new TimeSeriesInstance
     */
    public TimeSeriesInstance getHSlice(int dimensionToKeep) {
        return getHSlice(new int[]{dimensionToKeep});
    }


    /**
     * Returns a string containing:
     * num dimensions and class label index
     * then for each dimension: the series
     *
     * @return instance info, then for each dimension: series
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();

        sb.append("Num Dimensions: ").append(getNumDimensions()).append(" Class Label Index: ").append(labelIndex);
        for (TimeSeries ts : seriesDimensions) {
            sb.append(System.lineSeparator());
            sb.append(ts.toString());
        }

        return sb.toString();
    }

    /**
     * Returns a 2d array, containing all dimensions and series values.
     *
     * @return each dimension of series
     */
    public double[][] toValueArray() {
        double[][] output = new double[this.seriesDimensions.size()][];
        for (int i = 0; i < output.length; ++i) {
            //clone the data so the underlying representation can't be modified
            output[i] = seriesDimensions.get(i).toValueArray();
        }
        return output;
    }

    /**
     * Returns a transposed 2d array.
     *
     * @return transposed array
     */
    public double[][] toTransposedArray() {
        double[][] original = this.toValueArray();
        double[][] transposed = new double[maxLength][seriesDimensions.size()];

        // for each dimension
        for (int i = 0; i < seriesDimensions.size(); i++) {
            // for each value in series
            for (int j = 0; j < maxLength; j++) {
                transposed[j][i] = original[i][j];
            }
        }

        return transposed;
    }

    /**
     * Returns the TimeSeries object at the index.
     *
     * @param index to get
     * @return TimeSeries object
     */
    public TimeSeries get(int index) {
        return this.seriesDimensions.get(index);
    }

    /**
     * Returns the target value.
     *
     * @return target value
     */
    public double getTargetValue() {
        return targetValue;
    }

    /**
     * Returns an iterator, iterating over the each dimension of series.
     *
     * @return dimension iterator
     */
    @Override
    public Iterator<TimeSeries> iterator() {
        return seriesDimensions.iterator();
    }

    /**
     * Returns a Stream of TimeSeries objects.
     *
     * @return Stream of TimeSeries.
     */
    public Stream<TimeSeries> stream() {
        return seriesDimensions.stream();
    }

    /**
     * Returns a new TimeSeriesInstance object containing the dimensions from
     * the specified start, inclusive, and end, exclusive.
     *
     * @param startInclusive index of dimension to start from (inclusive)
     * @param endExclusive   index of dimension to end at (exclusive)
     * @return new TimeSeriesInstance object
     */
    public TimeSeriesInstance getHSlice(int startInclusive, int endExclusive) {
        // copy construct a new inst
        final TimeSeriesInstance tsi = copy();
        // trim current data to a subset
        tsi.seriesDimensions = seriesDimensions.subList(startInclusive, endExclusive);
        tsi.dataChecks();
        return tsi;
    }

    /**
     * Returns a 2d list, containing the dimensions from the specified start,
     * inclusive and end, exclusive.
     *
     * @param startInclusive index of dimension to start from (inclusive)
     * @param endExclusive   index of dimension to end at (exclusive)
     * @return 2d list
     */
    public List<List<Double>> getHSliceList(int startInclusive, int endExclusive) {
        return seriesDimensions.subList(startInclusive, endExclusive).stream().map(TimeSeries::getSeries).collect(Collectors.toList());
    }

    /**
     * Returns a 2d array, containing the dimensions from the specified start,
     * inclusive and end, exclusive.
     *
     * @param startInclusive index of dimension to start from (inclusive)
     * @param endExclusive   index of dimension to end at (exclusive)
     * @return 2d array
     */
    public double[][] getHSliceArray(int startInclusive, int endExclusive) {
        return getHSliceList(startInclusive, endExclusive).stream().map(dim -> dim.stream().mapToDouble(d -> d).toArray()).toArray(double[][]::new);
    }

    /**
     * Returns a 2d list, containing all dimensions but series cut between the
     * specified start, inclusive, and end, exclusive.
     *
     * @param startInclusive index to start from (inclusive)
     * @param endExclusive   index to end from (exclusive)
     * @return 2d list
     */
    public List<List<Double>> getVSliceList(int startInclusive, int endExclusive) {
        return seriesDimensions.stream().map(dim -> dim.getVSliceList(startInclusive, endExclusive)).collect(Collectors.toList());
    }

    /**
     * Returns a 2d array, containing all dimensions but series cut between the
     * specified start, inclusive, and end, exclusive.
     *
     * @param startInclusive index to start from (inclusive)
     * @param endExclusive   index to end from (exclusive)
     * @return 2d array
     */
    public double[][] getVSliceArray(int startInclusive, int endExclusive) {
        return getVSliceList(startInclusive, endExclusive).stream().map(dim -> dim.stream().mapToDouble(d -> d).toArray()).toArray(double[][]::new);
    }

    /**
     * Returns a new TimeSeriesInstance object, containing all dimensions but
     * series cut between the specified start, inclusive, and end, exclusive.
     *
     * @param startInclusive index to start from (inclusive)
     * @param endExclusive   index to end from (exclusive)
     * @return new TimeSeriesInstance object
     */
    public TimeSeriesInstance getVSlice(int startInclusive, int endExclusive) {
        // copy construct a new inst
        final TimeSeriesInstance tsi = copy();
        // trim current data to a subset
        tsi.seriesDimensions = seriesDimensions.stream().map(dim -> dim.getVSlice(startInclusive, endExclusive)).collect(Collectors.toList());
        tsi.dataChecks();
        return tsi;
    }

    /**
     * Returns whether a TimeSeriesInstance object is equal to another based if
     * label index is equal, target value is equal and series are equal.
     *
     * @param other object
     * @return true if equal, false if not
     */
    @Override
    public boolean equals(final Object other) {
        if (!(other instanceof TimeSeriesInstance)) {
            return false;
        }
        final TimeSeriesInstance that = (TimeSeriesInstance) other;
        return labelIndex == that.labelIndex &&
                Double.compare(that.targetValue, targetValue) == 0 &&
                seriesDimensions.equals(that.seriesDimensions);
    }

    /**
     * Returns an int of the hash code based on the series and label index.
     *
     * @return hash code
     */
    @Override
    public int hashCode() {
        return Objects.hash(seriesDimensions, labelIndex);
    }

    /**
     * Returns whether data has label index.
     *
     * @return true if label index is set, fasle if not
     */
    public boolean isLabelled() {
        // is labelled if label index points to a class label
        return labelIndex >= 0;
    }

    /**
     * Returns whether data is regressed.
     *
     * @return true if regressed, false if not
     */
    public boolean isRegressed() {
        // is regressed if the target value is set
        return !Double.isNaN(targetValue);
    }

    /**
     * Returns whether data is for a classification problem.
     *
     * @return true if classification problem, false if not
     */
    public boolean isClassificationProblem() {
        // if a set of class labels are set then it's a classification problem
        return labelIndex >= 0;
    }

    /**
     * Returns whether data is for a regression problem.
     *
     * @return true if regression problem, false if not
     */
    public boolean isRegressionProblem() {
        return !isClassificationProblem();
    }
}
