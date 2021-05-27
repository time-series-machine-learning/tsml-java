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

import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Data structure able to handle unequal length, unequally spaced, univariate or
 * multivariate time series.
 *
 * @author Aaron Bostrom, 2020
 */
public class TimeSeriesInstances implements Iterable<TimeSeriesInstance>, Serializable {

    /* Meta Information */
    private String description = "";
    private String problemName = "default";
    private boolean isEquallySpaced = true;
    private boolean hasMissing;
    private boolean isEqualLength;

    private boolean isMultivariate;
    private boolean hasTimeStamps;

    // this could be by dimension, so could be a list.
    private int minLength;
    private int maxLength;
    private int maxNumDimensions;

    /**
     * Returns the highest number of dimensions from the instances in the data.
     *
     * @return highest number of dimensions
     */
    public int getMaxNumDimensions() {
        return maxNumDimensions;
    }

    /**
     * Returns the problem name.
     *
     * @return String problem name
     */
    public String getProblemName() {
        return problemName;
    }

    /**
     * Returns whether the data has time stamps or not.
     *
     * @return boolean true if has time stamps, false if not
     */
    public boolean hasTimeStamps() {
        return hasTimeStamps;
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
     * Returns whether data is equally spaced.
     *
     * @return true if equally spaced, false if not
     */
    public boolean isEquallySpaced() {
        return isEquallySpaced;
    }

    /**
     * Returns whether data is multivariate.
     *
     * @return true if multivariate, false if not
     */
    public boolean isMultivariate() {
        return isMultivariate;
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

    /**
     * Returns the number of class labels in the data.
     *
     * @return number of class labels
     */
    public int numClasses() {
        return classLabels.length;
    }

    /**
     * Sets the problem name to the one passed.
     *
     * @param problemName to set
     */
    public void setProblemName(String problemName) {
        this.problemName = problemName;
    }

    /**
     * Returns the description of the data if set, or null if not.
     *
     * @return description of data
     */
    public String getDescription() {
        return description;
    }

    /**
     * Sets the description of the data.
     *
     * @param description of data
     */
    public void setDescription(String description) {
        this.description = description;
    }

    /* End Meta Information */

    private List<TimeSeriesInstance> seriesCollection = new ArrayList<>();

    // mapping for class labels. so ["apple","orange"] => [0,1]
    // this could be optional for example regression problems.
    public static String[] EMPTY_CLASS_LABELS = new String[0];
    private String[] classLabels = EMPTY_CLASS_LABELS;

    private int[] classCounts;

    public TimeSeriesInstances(final String[] classLabels) {
        this.classLabels = classLabels;

        minLength = Integer.MAX_VALUE;
        maxLength = 0;
        maxNumDimensions = 0;
    }

    public TimeSeriesInstances(final List<? extends List<? extends List<Double>>> rawData, List<Double> targetValues) {

        int index = 0;
        for (final List<? extends List<Double>> series : rawData) {
            //using the add function means all stats should be correctly counted.
            seriesCollection.add(new TimeSeriesInstance(series, targetValues.get(index++)));
        }

        dataChecks();
    }

    public TimeSeriesInstances(final List<? extends List<? extends List<Double>>> rawData, String[] classLabels, final List<Double> labelIndices) {
        this(rawData, labelIndices.stream().map(TimeSeriesInstance::discretiseLabelIndex).collect(Collectors.toList()), classLabels);
    }

    public TimeSeriesInstances(final List<? extends List<? extends List<Double>>> rawData, final List<Integer> labelIndexes, String[] classLabels) {

        this.classLabels = classLabels;

        int index = 0;
        for (final List<? extends List<Double>> series : rawData) {
            //using the add function means all stats should be correctly counted.
            seriesCollection.add(new TimeSeriesInstance(series, labelIndexes.get(index++).intValue()));
        }

        dataChecks();
    }

    public TimeSeriesInstances(double[][][] rawData, double[] labelIndices, String[] labels) {
        this(rawData, Arrays.stream(labelIndices).mapToInt(TimeSeriesInstance::discretiseLabelIndex).toArray(), labels);
    }

    public TimeSeriesInstances(double[][][] rawData, double[] targetValues) {

        int index = 0;
        for (double[][] series : rawData) {
            //using the add function means all stats should be correctly counted.
            seriesCollection.add(new TimeSeriesInstance(series, targetValues[index++]));
        }
    }

    public TimeSeriesInstances(final double[][][] rawData, int[] labelIndexes, String[] labels) {

        classLabels = labels;

        int index = 0;
        for (double[][] series : rawData) {
            //using the add function means all stats should be correctly counted.
            seriesCollection.add(new TimeSeriesInstance(series, labelIndexes[index++], classLabels));
        }

        dataChecks();
    }

    public TimeSeriesInstances(List<? extends TimeSeriesInstance> data) {
        this(data, EMPTY_CLASS_LABELS);
    }

    public TimeSeriesInstances(List<? extends TimeSeriesInstance> data, String[] classLabels) {

        this.classLabels = classLabels;

        seriesCollection.addAll(data);

        dataChecks();
    }

    public TimeSeriesInstances(TimeSeriesInstance[] data, String[] classLabels) {
        this(Arrays.asList(data), classLabels);
    }

    public TimeSeriesInstances(TimeSeriesInstance[] data) {
        this(Arrays.asList(data));
    }

    private void dataChecks() {

        if (seriesCollection == null) {
            throw new NullPointerException("no series collection");
        }
        if (classLabels == null) {
            throw new NullPointerException("no class labels");
        }

        calculateLengthBounds();
        calculateIfMissing();
        calculateIfMultivariate();
        calculateNumDimensions();
    }

    private void calculateClassCounts() {
        classCounts = new int[classLabels.length];
        for (TimeSeriesInstance inst : seriesCollection) {
            classCounts[inst.getLabelIndex()]++;
        }
    }

    private void calculateLengthBounds() {
        minLength = seriesCollection.stream().mapToInt(TimeSeriesInstance::getMinLength).min().orElse(-1);
        maxLength = seriesCollection.stream().mapToInt(TimeSeriesInstance::getMaxLength).max().orElse(-1);
        isEqualLength = minLength == maxLength;
    }

    private void calculateNumDimensions() {
        maxNumDimensions = seriesCollection.stream().mapToInt(TimeSeriesInstance::getNumDimensions).max().getAsInt();
    }

    private void calculateIfMultivariate() {
        isMultivariate = seriesCollection.stream().map(TimeSeriesInstance::isMultivariate).anyMatch(Boolean::booleanValue);
    }

    private void calculateIfMissing() {
        // if any of the instance have a missing value then this is true.
        hasMissing = seriesCollection.stream().map(TimeSeriesInstance::hasMissing).anyMatch(Boolean::booleanValue);
    }

    /**
     * Returns a String array containing the class labels.
     *
     * @return array containing class labels
     */
    public String[] getClassLabels() {
        return classLabels;
    }

    /**
     * Returns a string containing all of the class labels separated by a space.
     *
     * @return class labels formatted
     */
    public String getClassLabelsFormatted() {
        StringBuilder output = new StringBuilder(" ");

        for (String s : classLabels)
            output.append(s).append(" ");

        return output.toString();
    }

    /**
     * Returns an array containing the counter of each class.
     *
     * @return an array of class counts
     */
    public int[] getClassCounts() {
        calculateClassCounts();
        return classCounts;
    }

    /**
     * Adds a new TimeSeriesInstance to the data.
     *
     * @param newSeries to add
     */
    public void add(final TimeSeriesInstance newSeries) {
        seriesCollection.add(newSeries);

        //guard for if we're going to force update classCounts after.
        if (classCounts != null && newSeries.getLabelIndex() < classCounts.length)
            classCounts[newSeries.getLabelIndex()]++;
        
        if(seriesCollection.size() == 1) {
            // was empty / this is the first inst being added to the list
            // therefore metadata is set to -1's
            // need to change minLength to a very large number else the -1 invalid value is always the minimum length
            minLength = Integer.MAX_VALUE;
        }
        minLength = Math.min(newSeries.getMinLength(), minLength);
        maxLength = Math.max(newSeries.getMaxLength(), maxLength);
        maxNumDimensions = Math.max(newSeries.getNumDimensions(), maxNumDimensions);
        hasMissing |= newSeries.hasMissing();
        isEqualLength = minLength == maxLength;
        isMultivariate |= newSeries.isMultivariate();
    }

    /**
     * Returns a string containing:
     * class labels
     * then for each dimension:
     * - num dimensions, class label index
     * - the series
     *
     * @return class labels, then for instance: num dimensions, class label index
     * and the series
     */
    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder();

        sb.append("Labels: [").append(classLabels[0]);
        for (int i = 1; i < classLabels.length; i++) {
            sb.append(',');
            sb.append(classLabels[i]);
        }
        sb.append(']').append(System.lineSeparator());

        for (final TimeSeriesInstance series : seriesCollection) {
            sb.append(series.toString());
            sb.append(System.lineSeparator());
        }

        return sb.toString();
    }

    /**
     * Returns a 3d array, first index is which instance, 2nd index is which
     * dimension in the instance, 3rd index is the array of values in that dimension.
     *
     * @return values in 3d array format
     */
    public double[][][] toValueArray() {
        final double[][][] output = new double[seriesCollection.size()][][];
        for (int i = 0; i < output.length; ++i) {
            // clone the data so the underlying representation can't be modified
            output[i] = seriesCollection.get(i).toValueArray();
        }
        return output;
    }

    /**
     * Returns an array containing each class index.
     *
     * @return array of class indexes
     */
    public int[] getClassIndexes() {
        int[] out = new int[numInstances()];
        int index = 0;
        for (TimeSeriesInstance inst : seriesCollection) {
            out[index++] = inst.getLabelIndex();
        }
        return out;
    }

    /**
     * Returns an array containing the value(s) from each instance at the index
     * passed.
     *
     * Assumes equal number of dimensions.
     *
     * @param index to get data from
     * @return array of values at index
     */
    public double[] getVSliceArray(int index) {
        double[] out = new double[numInstances() * seriesCollection.get(0).getNumDimensions()];
        int i = 0;
        for (TimeSeriesInstance inst : seriesCollection) {
            for (TimeSeries ts : inst)
                // if the index isn't always valid, populate with NaN values.
                out[i++] = ts.hasValidValueAt(index) ? ts.getValue(index) : Double.NaN;
        }

        return out;
    }

    /**
     * Returns a 3d List containing the values at the indexes passes from each
     * instance, including all dimensions within each instance.
     *
     * @param indexesToKeep to get
     * @return 3d list of values at indexes passed
     */
    public List<List<List<Double>>> getVSliceList(int[] indexesToKeep) {
        return getVSliceList(Arrays.stream(indexesToKeep).boxed().collect(Collectors.toList()));
    }

    /**
     * Returns a 3d List containing the values at the indexes passes from each
     * instance, including all dimensions within each instance.
     *
     * @param indexesToKeep to get
     * @return 3d list of values at indexes passed
     */
    public List<List<List<Double>>> getVSliceList(List<Integer> indexesToKeep) {
        List<List<List<Double>>> out = new ArrayList<>(numInstances());
        for (TimeSeriesInstance inst : seriesCollection) {
            out.add(inst.getVSliceList(indexesToKeep));
        }

        return out;
    }

    /**
     * Returns a 3d array containing the values at the indexes passes from each
     * instance, including all dimensions within each instance.
     *
     * @param indexesToKeep to get
     * @return 3d array of values at indexes passed
     */
    public double[][][] getVSliceArray(int[] indexesToKeep) {
        return getVSliceArray(Arrays.stream(indexesToKeep).boxed().collect(Collectors.toList()));
    }

    /**
     * Returns a 3d array containing the values at the indexes passes from each
     * instance, including all dimensions within each instance.
     *
     * @param indexesToKeep to get
     * @return 3d array of values at indexes passed
     */
    public double[][][] getVSliceArray(List<Integer> indexesToKeep) {
        double[][][] out = new double[numInstances()][][];
        int i = 0;
        for (TimeSeriesInstance inst : seriesCollection) {
            out[i++] = inst.getVSliceArray(indexesToKeep);
        }

        return out;
    }

    /**
     * Returns a 2d array containing the values at each instance at the dimension
     * passed. e.g. passing '2' will give the values from each instance at the 3rd
     * dimension.
     *
     * Assumes equal number of dimensions.
     *
     * @param dimensionToKeep to get
     * @return 2d array of values
     */
    public double[][] getHSliceArray(int dimensionToKeep) {
        double[][] out = new double[numInstances()][];
        int i = 0;
        for (TimeSeriesInstance inst : seriesCollection) {
            // if the index isn't always valid, populate with NaN values.
            out[i++] = inst.getHSliceArray(dimensionToKeep);
        }
        return out;
    }

    /**
     * Returns a 3d list containing the values for each instance, at the
     * dimensions passed. e.g. '[0, 1]' would return the values for every instance
     * at the first and second dimensions.
     *
     * @param dimensionsToKeep to get
     * @return 3d list of values
     */
    public List<List<List<Double>>> getHSliceList(int[] dimensionsToKeep) {
        return getHSliceList(Arrays.stream(dimensionsToKeep).boxed().collect(Collectors.toList()));
    }

    /**
     * Returns a 3d list containing the values for each instance, at the
     * dimensions passed. e.g. '[0, 1]' would return the values for every instance
     * at the first and second dimensions.
     *
     * @param dimensionsToKeep to get
     * @return 3d list of values
     */
    public List<List<List<Double>>> getHSliceList(List<Integer> dimensionsToKeep) {
        List<List<List<Double>>> out = new ArrayList<>(numInstances());
        for (TimeSeriesInstance inst : seriesCollection) {
            out.add(inst.getHSliceList(dimensionsToKeep));
        }

        return out;
    }

    /**
     * Returns a 3d array containing the values for each instance, at the
     * dimensions passed. e.g. '[0, 1]' would return the values for every instance
     * at the first and second dimensions.
     *
     * @param dimensionsToKeep to get
     * @return 3d array of values
     */
    public double[][][] getHSliceArray(int[] dimensionsToKeep) {
        return getHSliceArray(Arrays.stream(dimensionsToKeep).boxed().collect(Collectors.toList()));
    }

    /**
     * Returns a 3d array containing the values for each instance, at the
     * dimensions passed. e.g. '[0, 1]' would return the values for every instance
     * at the first and second dimensions.
     *
     * @param dimensionsToKeep to get
     * @return 3d array of values
     */
    public double[][][] getHSliceArray(List<Integer> dimensionsToKeep) {
        double[][][] out = new double[numInstances()][][];
        int i = 0;
        for (TimeSeriesInstance inst : seriesCollection) {
            out[i++] = inst.getHSliceArray(dimensionsToKeep);
        }

        return out;
    }

    /**
     * Returns the TimeSeriesInstance at the index given.
     *
     * @param index to get
     * @return TimeSeriesInstance at index
     */
    public TimeSeriesInstance get(final int index) {
        return seriesCollection.get(index);
    }

    /**
     * Returns all of the TimeSeriesInstance objects inside.
     *
     * @return list of TimeSeriesInstance
     */
    public List<TimeSeriesInstance> getAll() {
        return seriesCollection;
    }

    /**
     * Returns the number of instances inside.
     *
     * @return number of instances.
     */
    public int numInstances() {
        return seriesCollection.size();
    }

    /**
     * Creates a Histogram of lengths of the time series
     *
     * @return Histogram of lengths
     */
    public Map<Integer, Integer> getHistogramOfLengths() {
        Map<Integer, Integer> out = new TreeMap<>();
        for (TimeSeriesInstance inst : seriesCollection) {
            for (TimeSeries ts : inst) {
                out.merge(ts.getSeriesLength(), 1, Integer::sum);
            }
        }

        return out;
    }

    /**
     * Returns an iterator, iterating over the each instance.
     *
     * @return instance iterator
     */
    @Override
    public Iterator<TimeSeriesInstance> iterator() {
        return seriesCollection.iterator();
    }

    /**
     * Returns a Stream of TimeSeriesInstance objects.
     *
     * @return Stream of TimeSeries.
     */
    public Stream<TimeSeriesInstance> stream() {
        return seriesCollection.stream();
    }

    /**
     * Returns a 3d List containing the values, for each instance, for each dimension,
     * between the start (inclusive) and end (exclusive).
     *
     * @param startInclusive to start at (inclusive)
     * @param endExclusive to end at (exclusive)
     * @return 3d list of values
     */
    public List<List<List<Double>>> getVSliceList(int startInclusive, int endExclusive) {
        return seriesCollection.stream().map(inst -> inst.getVSliceList(startInclusive, endExclusive)).collect(Collectors.toList());
    }

    /**
     * Returns a TimeSeriesInstances object containing instances with values,
     * between the start (inclusive) and end (exclusive).
     *
     * @param startInclusive to start at (inclusive)
     * @param endExclusive to end at (exclusive)
     * @return TimeSeriesInstances object
     */
    public TimeSeriesInstances getVSlice(int startInclusive, int endExclusive) {
        final TimeSeriesInstances tsi = new TimeSeriesInstances(classLabels);
        tsi.seriesCollection = seriesCollection.stream().map(inst -> inst.getVSlice(startInclusive, endExclusive)).collect(Collectors.toList());
        tsi.dataChecks();
        return tsi;
    }

    /**
     * Returns a 3d array containing the values, for each instance, for each dimension,
     * between the start (inclusive) and end (exclusive).
     *
     * @param startInclusive to start at (inclusive)
     * @param endExclusive to end at (exclusive)
     * @return 3d array of values
     */
    public double[][][] getVSliceArray(int startInclusive, int endExclusive) {
        return seriesCollection.stream().map(inst -> inst.getVSliceArray(startInclusive, endExclusive)).toArray(double[][][]::new);
    }

    /**
     * Returns a 3d list containing the values, for each instance, from the start
     * dimension (inclusive) and end dimension (exclusive) given.
     *
     * @param startInclusive to start at (inclusive)
     * @param endExclusive to end at (exclusive)
     * @return 3d list of values
     */
    public List<List<List<Double>>> getHSliceList(int startInclusive, int endExclusive) {
        return seriesCollection.stream().map(inst -> inst.getHSliceList(startInclusive, endExclusive)).collect(Collectors.toList());
    }

    /**
     * Returns a 3d array containing the values, for each instance, from the start
     * dimension (inclusive) and end dimension (exclusive) given.
     *
     * @param startInclusive to start at (inclusive)
     * @param endExclusive to end at (exclusive)
     * @return 3d array of values
     */
    public double[][][] getHSliceArray(int startInclusive, int endExclusive) {
        return seriesCollection.stream().map(inst -> inst.getHSliceArray(startInclusive, endExclusive)).toArray(double[][][]::new);
    }

    /**
     * Returns a TimeSeriesInstances object containing instances with values,
     * from the start dimension (inclusive) and end dimension (exclusive)
     *
     * @param startInclusive to start at (inclusive)
     * @param endExclusive to end at (exclusive)
     * @return TimeSeriesInstances object
     */
    public TimeSeriesInstances getHSlice(int startInclusive, int endExclusive) {
        final TimeSeriesInstances tsi = new TimeSeriesInstances(classLabels);
        tsi.seriesCollection = seriesCollection.stream().map(inst -> inst.getHSlice(startInclusive, endExclusive)).collect(Collectors.toList());
        tsi.dataChecks();
        return tsi;
    }

    /**
     * Function to pad data up (inclusive) maxLength with '0' values.
     *
     * @param newMaxLength the length to pad to
     * @return padded TimeSeriesInstances object
     */
    public TimeSeriesInstances padWithZerosOrMissing(final int newMaxLength) {
        double[][][] temp = new double[numInstances()][getMaxNumDimensions()][newMaxLength];

        // for each TimeSeriesInstance
        for (int i = 0; i < seriesCollection.size(); i++) {
            TimeSeriesInstance tsi = seriesCollection.get(i);

            // for each dimension within the TimeSeriesInstance
            for (int j = 0; j < tsi.getNumDimensions(); j++) {
                // copy over values from series
                double[] series = tsi.get(j).toValueArray();
                System.arraycopy(series, 0, temp[i][j], 0, series.length);
            }
        }

        TimeSeriesInstances padded = new TimeSeriesInstances(temp, getClassIndexes(), getClassLabels());
        padded.setProblemName(getProblemName());
        padded.setDescription(getDescription());

        return padded;
    }

    /**
     * Returns whether a TimeSeriesInstances object is equal to another based if
     * series collection is the exact same and the class labels are the exact same.
     *
     * @param other object
     * @return true if equal, false if not
     */
    @Override
    public boolean equals(final Object other) {
        if (!(other instanceof TimeSeriesInstances)) {
            return false;
        }
        final TimeSeriesInstances that = (TimeSeriesInstances) other;
        return Objects.equals(seriesCollection, that.seriesCollection) &&
                Arrays.equals(classLabels, that.classLabels);
    }

    /**
     * Returns an int of the hash code based on the series collection and class
     * labels.
     *
     * @return hash code
     */
    @Override
    public int hashCode() {
        return Objects.hash(seriesCollection, classLabels);
    }

    /**
     * Returns whether data is for a classification problem.
     *
     * @return true if classification problem, false if not
     */
    public boolean isClassificationProblem() {
        // if a set of class labels are set then it's a classification problem
        return classLabels.length >= 0;
    }

    /**
     * Returns whether data is for a regression problem.
     *
     * @return true if regression problem, false if not
     */
    public boolean isRegressionProblem() {
        return !isClassificationProblem();
    }

    /**
     * Get inst index binned by class.
     * I.e. class 1 contains {3,5,6} and class 2 contains {1,2,4}
     *
     * @return
     */
    public List<List<Integer>> getInstIndicesByClass() {
        final List<List<Integer>> bins = new ArrayList<>(numClasses());
        for(int i = 0; i < numClasses(); i++) {
            bins.add(new ArrayList<>());
        }
        for(int i = 0; i < numInstances(); i++) {
            final TimeSeriesInstance inst = get(i);
            final int labelIndex = inst.getLabelIndex();
            final List<Integer> bin = bins.get(labelIndex);
            bin.add(i);
        }
        return bins;
    }
    
    public List<TimeSeriesInstances> getInstsByClass() {
        return getInstIndicesByClass().stream().map(
                indices -> new TimeSeriesInstances(
                        indices.stream().map(this::get).collect(Collectors.toList()),
                        getClassLabels())).collect(Collectors.toList());
    }
    
    public void addAll(Iterable<TimeSeriesInstance> insts) {
        for(TimeSeriesInstance inst : insts) {
            add(inst);
        }
    }
}
