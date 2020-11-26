/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */   
package tsml.transformers;

import tsml.classifiers.TrainTimeContractable;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.transformers.shapelet_tools.OrderLineObj;
import tsml.transformers.shapelet_tools.Shapelet;
import tsml.transformers.shapelet_tools.ShapeletCandidate;
import tsml.transformers.shapelet_tools.ShapeletTransformTimingUtilities;
import tsml.transformers.shapelet_tools.class_value.NormalClassValue;
import tsml.transformers.shapelet_tools.distance_functions.ShapeletDistance;
import tsml.transformers.shapelet_tools.quality_measures.ShapeletQuality;
import tsml.transformers.shapelet_tools.quality_measures.ShapeletQuality.ShapeletQualityChoice;
import tsml.transformers.shapelet_tools.search_functions.ShapeletSearch;
import tsml.transformers.shapelet_tools.search_functions.ShapeletSearchFactory;
import tsml.transformers.shapelet_tools.search_functions.ShapeletSearchOptions;
import utilities.NumUtils;
import utilities.class_counts.ClassCounts;
import utilities.rescalers.SeriesRescaler;
import weka.core.*;

import java.io.*;
import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * 
 * NOTE: As shapelet extraction can be time consuming, there is an option to
 * output shapelets to a text file (Default location is in the root dir of the
 * project, file name "defaultShapeletOutput.txt").
 *
 * Default settings are TO NOT PRODUCE OUTPUT FILE - unless file name is
 * changed, each successive filter will overwrite the output (see
 * "setLogOutputFile(String fileName)" to change file dir and name).
 *
 * To reconstruct a filter from this output, please see the method
 * "createFilterFromFile(String fileName)".
 * 
 * 
 * 
 * A filter to transform a dataset by k shapelets. Once built on a training set,
 * the filter can be used to transform subsequent datasets using the extracted
 * shapelets.
 * <p>
 * See <a href=
 * "http://delivery.acm.org/10.1145/2340000/2339579/p289-lines.pdf?ip=139.222.14.198&acc=ACTIVE%20SERVICE&CFID=221649628&CFTOKEN=31860141&__acm__=1354814450_3dacfa9c5af84445ea2bfd7cc48180c8">
 * Lines J., Davis, L., Hills, J., Bagnall, A.: A shapelet transform for time
 * series classification. In: Proc. 18th ACM SIGKDD (2012)</a>
 *
 * @author Jason Lines, Aaron Bostrom and Tony Bagnall
 *
 *         Refactored version for
 */
public class ShapeletTransform implements Serializable, TechnicalInformationHandler, TrainableTransformer {
    // Global defaults. Max should be a lambda set to series length
    public final static int MAXTRANSFORMSIZE = 1000;
    public final static int DEFAULT_MINSHAPELETLENGTH = 3;
    public final static int DEFAULT_MAXSHAPELETLENGTH = 23;

    // Im not sure this is used anywhere, should be in Options for condensing data?
    public static final int minimumRepresentation = 25; // If condensing the search set, this is the minimum number of
                                                        // instances per class to search

    // Variables for experiments
    protected static long subseqDistOpCount;
    private boolean removeSelfSimilar = true;
    private boolean pruneMatchingShapelets;

    // this int is used to serialise our position when iterating through a dataset.
    public int casesSoFar;
    public boolean searchComplete = false;

    protected boolean supressOutput = true; // defaults to print in System.out AS WELL as file, set to true to stop
                                            // printing to console
    protected int numShapelets; // The maximum number of shapelets in the transform. This is K and is different
                                // to the total number of shapelets to look for/looked for

    protected ArrayList<Shapelet> shapelets;
    protected String ouputFileLocation = "defaultShapeletOutput.txt"; // default store location
    protected boolean recordShapelets; // default action is to write an output file
    protected long numShapeletsEvaluated = 0;// This counts the total number of shapelets returned by
                                             // searchForShapeletsInSeries. It does not include early abandoned
                                             // shapelets
    protected long numEarlyAbandons = 0;// This counts number of shapelets early abandoned

    // All of these can be in an ShapeletTransformOptions
    // ShapeletTransformFactoryOptions.ShapeletTransformOptions options;
    protected boolean roundRobin;
    protected transient ShapeletQuality quality;
    protected boolean useCandidatePruning;
    protected boolean useRoundRobin;
    protected boolean useBalancedClasses;
    protected ShapeletDistance shapeletDistance;
    protected ShapeletSearch searchFunction;

    protected Comparator<Shapelet> shapeletComparator;
    protected NormalClassValue classValue;
    protected String serialName;
    protected Shapelet worstShapelet;
    
    protected Instances inputData;
    protected TimeSeriesInstances inputDataTS;

    protected ArrayList<Shapelet> kShapelets;
    protected int candidatePruningStartPercentage;
    protected long count;
    protected Map<Double, ArrayList<Shapelet>> kShapeletsMap;
    protected static final double ROUNDING_ERROR_CORRECTION = 0.000000000000001;
    protected int[] dataSourceIDs;
    /**
     * Contract data
     */
    protected boolean adaptiveTiming = false;
    private double timePerShapelet = 0;
    private long totalShapeletsPerSeries = 0; // Found once
    private long shapeletsSearchedPerSeries = 0;// This is taken from the search function
    private int numSeriesToUse = 0;
    private long contractTime = 0; // nano seconds time. If set to zero everything reverts to
                                   // BalancedClassShapeletTransform
    private double beta = 0.2;

    /**
     * Default constructor; Quality measure defaults to information gain.
     */
    public ShapeletTransform() {
        this(MAXTRANSFORMSIZE, DEFAULT_MINSHAPELETLENGTH, DEFAULT_MAXSHAPELETLENGTH,
                ShapeletQualityChoice.INFORMATION_GAIN);
    }

    /**
     * Constructor for generating a shapelet transform from an ArrayList of
     * Shapelets.
     *
     * @param shapes
     */
    public ShapeletTransform(ArrayList<Shapelet> shapes) {
        this();
        this.shapelets = shapes;
        this.numShapelets = shapelets.size();
    }

    /**
     * Full constructor to create a usable filter. Quality measure defaults to
     * information gain.
     *
     * @param k                 the number of shapelets to be generated
     * @param minShapeletLength minimum length of shapelets
     * @param maxShapeletLength maximum length of shapelets
     */
    public ShapeletTransform(int k, int minShapeletLength, int maxShapeletLength) {
        this(k, minShapeletLength, maxShapeletLength, ShapeletQualityChoice.INFORMATION_GAIN);

    }

    /**
     * Full, exhaustive, constructor for a filter. Quality measure set via enum,
     * invalid selection defaults to information gain.
     *
     * @param k                 the number of shapelets to be generated
     * @param minShapeletLength minimum length of shapelets
     * @param maxShapeletLength maximum length of shapelets
     * @param qualityChoice     the shapelet quality measure to be used with this
     *                          filter
     */
    public ShapeletTransform(int k, int minShapeletLength, int maxShapeletLength, ShapeletQualityChoice qualityChoice) {
        this.numShapelets = k;
        this.shapelets = new ArrayList<>();
        this.useCandidatePruning = false;
        this.casesSoFar = 0;
        this.recordShapelets = true; // default action is to write an output file
        this.roundRobin = false;
        this.useRoundRobin = false;
        this.shapeletComparator = new Shapelet.LongOrder();
        this.kShapelets = new ArrayList<>();

        setQualityMeasure(qualityChoice);
        this.shapeletDistance = new ShapeletDistance();
        this.classValue = new NormalClassValue();

        ShapeletSearchOptions sOp = new ShapeletSearchOptions.Builder().setMin(minShapeletLength)
                .setMax(maxShapeletLength).build();
        this.searchFunction = new ShapeletSearchFactory(sOp).getShapeletSearch();
    }

    protected void initQualityBound(ClassCounts classDist) {
        if (!useCandidatePruning)
            return;
        quality.initQualityBound(classDist, candidatePruningStartPercentage);
    }

    /**
     * Sets the format of the filtered instances that are output. I.e. will include
     * k attributes each shapelet distance and a class value
     *
     * @param inputFormat the format of the input data
     * @return a new Instances object in the desired output format
     */
    @Override
    public Instances determineOutputFormat(Instances inputFormat) throws IllegalArgumentException {
        if (this.numShapelets < 1) {
            System.out.println(this.numShapelets);
            throw new IllegalArgumentException(
                    "ShapeletTransform not initialised correctly - please specify a value of k (this.numShapelets) that is greater than or equal to 1. It is currently set tp "
                            + this.numShapelets);
        }
        // Set up instances size and format.
        // int length = this.numShapelets;
        int length = this.shapelets.size();
        ArrayList<Attribute> atts = new ArrayList<>();
        String name;
        for (int i = 0; i < length; i++) {
            name = "Shapelet_" + i;
            atts.add(new Attribute(name));
        }
        if (inputFormat.classIndex() >= 0) {
            // Classification set, set class
            // Get the class values as a fast vector
            Attribute target = inputFormat.attribute(inputFormat.classIndex());

            FastVector vals = new FastVector(target.numValues());
            for (int i = 0; i < target.numValues(); i++) {
                vals.addElement(target.value(i));
            }
            atts.add(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(), vals));
        }
        Instances result = new Instances("Shapelets" + inputFormat.relationName(), atts, inputFormat.numInstances());
        if (inputFormat.classIndex() >= 0) {
            result.setClassIndex(result.numAttributes() - 1);
        }
        return result;
    }

    @Override
    public void fit(Instances data) {
        inputData = data;

        totalShapeletsPerSeries = ShapeletTransformTimingUtilities.calculateNumberOfShapelets(1,
                data.numAttributes() - 1, searchFunction.getMinShapeletLength(), searchFunction.getMaxShapeletLength());

        // check the input data is correct and assess whether the filter has been setup
        // correctly.
        trainShapelets(data);
        searchComplete = true;
        // we log the count from the subsequence distance before we reset it in the
        // transform.
        // we only care about the count from the train. What is it counting?
        count = shapeletDistance.getCount();
    }

    @Override
    public Instance transform(Instance data) {
        throw new UnsupportedOperationException(
                "NOT IMPLEMENTED YET. Cannot transform a single instance yet, is trivial though (ShapeletTransform.transform");
        // return buildTansformedDataset(data);
    }

    @Override
    public Instances transform(Instances data) throws IllegalArgumentException {
        return buildTansformedDataset(data);
    }

    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        // init out data for transforming.
        shapeletDistance.init(inputData);
        // setup classsValue
        classValue.init(inputData);

        Shapelet s;
        //get distance to each shapelet and create new instance
        int size = shapelets.size();

        //1 dimensional 
        double[][] out = new double[1][size];

        for (int i = 0; i < size; i++) {
            s = shapelets.get(i);
            shapeletDistance.setShapelet(s);
            out[0][i] = shapeletDistance.calculate(inst, 0);
        }
    
        return new TimeSeriesInstance(out, inst.getLabelIndex(), inst.getClassLabels());
    }

    @Override
    public TimeSeriesInstances transform(TimeSeriesInstances data) {
        return buildTansformedDataset(data);
    }

    @Override
    public void fit(TimeSeriesInstances data) {
        inputDataTS = data;

        totalShapeletsPerSeries = ShapeletTransformTimingUtilities.calculateNumberOfShapelets(1,
                data.getMaxLength(), searchFunction.getMinShapeletLength(), searchFunction.getMaxShapeletLength());

        // check the input data is correct and assess whether the filter has been setup
        // correctly.
        trainShapelets(data);
        searchComplete = true;
        // we log the count from the subsequence distance before we reset it in the
        // transform.
        // we only care about the count from the train. What is it counting?
        count = shapeletDistance.getCount();
    }

    protected void trainShapelets(Instances data) {
        // we might round robin the data in here.
        // So we need to override the input data with the new ordering.
        // we don't need to undo the roundRobin because we clone the data into a
        // different order.
        inputData = orderDataSet(data);
        // Initialise search function for ShapeletSearch object
        searchFunction.setComparator(shapeletComparator);
        searchFunction.init(inputData);
        // setup shapelet distance function (sDist). Just initialises the count to 0
        shapeletDistance.init(inputData);
        // setup classValue
        classValue.init(inputData);
        outputPrint("num shapelets before search " + numShapelets);
        // Contract is controlled by restricting number of shapelets per series.
        shapeletsSearchedPerSeries = searchFunction.getNumShapeletsPerSeries();
        shapelets = findBestKShapelets(inputData); // get k shapelets
        outputPrint(shapelets.size() + " Shapelets have been generated num shapelets now " + numShapelets);

    }

    protected void trainShapelets(TimeSeriesInstances data) {
         // we might round robin the data in here.
        // So we need to override the input data with the new ordering.
        // we don't need to undo the roundRobin because we clone the data into a
        // different order.
        inputDataTS = orderDataSet(data);
        // Initialise search function for ShapeletSearch object
        searchFunction.setComparator(shapeletComparator);
        searchFunction.init(inputDataTS);
        // setup shapelet distance function (sDist). Just initialises the count to 0
        shapeletDistance.init(inputDataTS);
        // setup classValue
        classValue.init(inputDataTS);
        outputPrint("num shapelets before search " + numShapelets);
        // Contract is controlled by restricting number of shapelets per series.
        shapeletsSearchedPerSeries = searchFunction.getNumShapeletsPerSeries();
        shapelets = findBestKShapelets(inputDataTS); // get k shapelets
        outputPrint(shapelets.size() + " Shapelets have been generated num shapelets now " + numShapelets);

    }

    /**
     * This method determines the order in which the series will be searched for
     * shapelets There are currently just two options: the original order or round
     * robin (take one of each class in turn). Round robin clones the data, which
     * could be an issue for resources but makes sense
     *
     * @param data
     * @return
     */
    private Instances orderDataSet(Instances data) {

        int dataSize = data.numInstances();
        // shapelets discovery has not yet been carried out, so this must be training
        // data
        dataSourceIDs = new int[dataSize];

        Instances dataset = data;
        if (roundRobin) {
            // Reorder the data in round robin order
            dataset = roundRobinData(data, dataSourceIDs);
        } else {
            for (int i = 0; i < dataSize; i++) {
                dataSourceIDs[i] = i;
            }
        }

        return dataset;
    }


        /**
     * This method determines the order in which the series will be searched for
     * shapelets There are currently just two options: the original order or round
     * robin (take one of each class in turn). Round robin clones the data, which
     * could be an issue for resources but makes sense
     *
     * @param data
     * @return
     */
    private TimeSeriesInstances orderDataSet(TimeSeriesInstances data) {

        int dataSize = data.numInstances();
        // shapelets discovery has not yet been carried out, so this must be training
        // data
        dataSourceIDs = new int[dataSize];

        TimeSeriesInstances dataset = data;
        if (roundRobin) {
            // Reorder the data in round robin order
            dataset = roundRobinData(data, dataSourceIDs);
        } else {
            for (int i = 0; i < dataSize; i++) {
                dataSourceIDs[i] = i;
            }
        }

        return dataset;
    }

    public TimeSeriesInstances buildTansformedDataset(TimeSeriesInstances data) {
        // init out data for transforming.
        shapeletDistance.init(inputData);
        // setup classsValue
        classValue.init(inputData);

        Shapelet s;
        // for each data, get distance to each shapelet and create new instance
        int size = shapelets.size();
        int dataSize = data.numInstances();

        //1 dimensional 
        double[][][] out = new double[dataSize][1][size];

        double dist;
        for (int i = 0; i < size; i++) {
            s = shapelets.get(i);
            shapeletDistance.setShapelet(s);

            for (int j = 0; j < dataSize; j++) {
                dist = shapeletDistance.calculate(data.get(j), j);
                out[j][0][i] = dist;
            }
        }

        return new TimeSeriesInstances(out, data.getClassIndexes(), data.getClassLabels());
    }

    // given a set of instances transform it by the internal shapelets.
    public Instances buildTansformedDataset(Instances data) {
        // Reorder the training data and reset the shapelet indexes
        Instances output = determineOutputFormat(data);

        // init out data for transforming.
        shapeletDistance.init(inputData);
        // setup classsValue
        classValue.init(inputData);

        Shapelet s;
        // for each data, get distance to each shapelet and create new instance
        int size = shapelets.size();
        int dataSize = data.numInstances();

        // create our data instances
        for (int j = 0; j < dataSize; j++) {
            output.add(new DenseInstance(size + 1));
        }

        double dist;
        for (int i = 0; i < size; i++) {
            s = shapelets.get(i);
            shapeletDistance.setShapelet(s);

            for (int j = 0; j < dataSize; j++) {
                dist = shapeletDistance.calculate(data.instance(j), j);
                output.instance(j).setValue(i, dist);
            }
        }

        // do the classValues.
        for (int j = 0; j < dataSize; j++) {
            // we always want to write the true ClassValue here. Irrelevant of binarised or
            // not.
            output.instance(j).setValue(size, data.instance(j).classValue());
        }

        return output;
    }

    /**
     * protected method for extracting k shapelets. this method extracts shapelets
     * series by series, using the searchFunction method searchForShapeletsInSeries,
     * which itself uses checkCandidate 1. The search method determines the method
     * of choosing shapelets. By default all are evaluated (ShapeletSearch) or the
     * alternative RandomSearch, which finds a fixed number of shapelets determined
     * by the time contract. 2. The qualityFunction assesses each candidate, and
     * uses the worstShapelet (set in checkCandidate) to test for inclusion and any
     * lower bounding. I dont think it uses it to test for inclusion. 3. self
     * similar are removed by default, and the method combine is used to merge the
     * current candidates and the new ones
     * 
     * @param data the data that the shapelets will be taken from
     * @return an ArrayList of FullShapeletTransform objects in order of their
     *         fitness (by infoGain, seperationGap then shortest length)
     */
    public ArrayList<Shapelet> findBestKShapelets(Instances data) {
        if (useBalancedClasses)
            return findBestKShapeletsBalanced(data);
        else
            return findBestKShapeletsOriginal(data);
    }

    public ArrayList<Shapelet> findBestKShapelets(TimeSeriesInstances data) {
        if (useBalancedClasses)
            return findBestKShapeletsBalanced(data);
        else
            return findBestKShapeletsOriginal(data);
    }

    /**
     *
     * @param data
     * @return
     */
    private ArrayList<Shapelet> findBestKShapeletsBalanced(TimeSeriesInstances data) {
        // If the number of shapelets we can calculate exceeds the total number in the
        // series, we will revert to full search
        ShapeletSearch full = new ShapeletSearch(searchFunction.getOptions());
        ShapeletSearch current = searchFunction;
        boolean contracted = true;
        boolean keepGoing = true;
        if (contractTime == 0) {
            contracted = false;
        }
        long startTime = System.nanoTime();
        long usedTime = 0;
        // This can be used to reduce the number of series searched. Better done with
        // Aaron's Condenser
        int numSeriesToUse = data.numInstances();
        // temp store of shapelets for each time series
        ArrayList<Shapelet> seriesShapelets;
        // construct a map for our K-shapelets lists, on for each classVal.
        if (kShapeletsMap == null) {
            kShapeletsMap = new TreeMap();
            for (int i = 0; i < data.numClasses(); i++) {
                kShapeletsMap.put((double) i, new ArrayList<>());
            }
        }
        // found out how many shapelets we want from each class, split evenly.
        int proportion = numShapelets / kShapeletsMap.keySet().size();

        outputPrint("Processing data for numShapelets " + numShapelets + " with proportion per class = " + proportion);
        outputPrint("in contract balanced: Contract (secs)" + contractTime / 1000000000.0);
        long prevEarlyAbandons = 0;
        int passes = 0;

        // continue processing series until we run out of time (if contracted)
        while (casesSoFar < numSeriesToUse && keepGoing) {
            // outputPrint("BALANCED: "+casesSoFar +" Cumulative time (secs) =
            // "+usedTime/1000000000.0+" Contract time (secs) ="+contractTime/1000000000.0+"
            // contracted = "+contracted+" search type = "+searchFunction.getSearchType());
            // get the Shapelets list based on the classValue of our current time series.
            kShapelets = kShapeletsMap.get(data.get(casesSoFar).getLabelIndex());
            // we only want to pass in the worstKShapelet if we've found K shapelets. but we
            // only care about
            // this class values worst one. This is due to the way we represent each classes
            // shapelets in the map.
            worstShapelet = kShapelets.size() == proportion ? kShapelets.get(kShapelets.size() - 1) : null;

            // set the series we're working with.
            shapeletDistance.setSeries(casesSoFar);
            // set the class value of the series we're working with.
            classValue.setShapeletValue(data.get(casesSoFar));
            long t1 = System.nanoTime();
            seriesShapelets = current.searchForShapeletsInSeries(data.get(casesSoFar), this::checkCandidate);
            long t2 = System.nanoTime();
            numShapeletsEvaluated += seriesShapelets.size();

            if (adaptiveTiming && contracted && passes == 0) {
                long tempEA = numEarlyAbandons - prevEarlyAbandons;
                prevEarlyAbandons = numEarlyAbandons;
                double newTimePerShapelet = (double) (t2 - t1) / (seriesShapelets.size() + tempEA);
                if (totalShapeletsPerSeries < (seriesShapelets.size() + tempEA))// Switch to full enum for next
                                                                                // iteration
                    current = full;
                else
                    current = searchFunction;

                System.out.println(" time for case " + casesSoFar + " evaluate  " + seriesShapelets.size()
                        + " but what about early ones?");
                System.out.println(" Est time per shapelet  " + timePerShapelet / 1000000000 + " actual "
                        + newTimePerShapelet / 1000000000);
                shapeletsSearchedPerSeries = adjustNumberPerSeries(contractTime - usedTime, numSeriesToUse - casesSoFar,
                        newTimePerShapelet);
                System.out.println("Changing number of shapelets sampled from "
                        + searchFunction.getNumShapeletsPerSeries() + " to " + shapeletsSearchedPerSeries);
                searchFunction.setNumShapeletsPerSeries(shapeletsSearchedPerSeries);
                System.out.println("data : " + casesSoFar + " has " + seriesShapelets.size() + " candidates"
                        + " cumulative early abandons " + numEarlyAbandons + " worst so far =" + worstShapelet
                        + " evaluated this series = " + (seriesShapelets.size() + tempEA));
            }
            if (seriesShapelets != null) {
                Collections.sort(seriesShapelets, shapeletComparator);
                if (isRemoveSelfSimilar())
                    seriesShapelets = removeSelfSimilar(seriesShapelets);

                kShapelets = combine(proportion, kShapelets, seriesShapelets);
            }

            // re-update the list because it's changed now.
            kShapeletsMap.put((double)data.get(casesSoFar).getLabelIndex(), kShapelets);
            casesSoFar++;
            createSerialFile();
            usedTime = System.nanoTime() - startTime;
            // Logic is we have underestimated the contract so can run back through. If we
            // over estimate it we will just stop.
            if (contracted) {
                if (casesSoFar == numSeriesToUse - 1 && !searchFunction.getSearchType().equals("FULL")) { /// HORRIBLE!
                    casesSoFar = 0;
                    passes++;
                }
                if (usedTime > contractTime)
                    keepGoing = false;
            }
        }

        kShapelets = buildKShapeletsFromMap(kShapeletsMap);

        this.numShapelets = kShapelets.size();

        if (recordShapelets)
            recordShapelets(kShapelets, this.ouputFileLocation);
        // if (!supressOutput)
        // writeShapelets(kShapelets, new OutputStreamWriter(System.out));

        return kShapelets;
    }

    private ArrayList<Shapelet> findBestKShapeletsBalanced(Instances data) {
        // If the number of shapelets we can calculate exceeds the total number in the
        // series, we will revert to full search
        ShapeletSearch full = new ShapeletSearch(searchFunction.getOptions());
        ShapeletSearch current = searchFunction;
        boolean contracted = true;
        boolean keepGoing = true;
        if (contractTime == 0) {
            contracted = false;
        }
        long startTime = System.nanoTime();
        long usedTime = 0;
        // This can be used to reduce the number of series searched. Better done with
        // Aaron's Condenser
        int numSeriesToUse = data.numInstances();
        // temp store of shapelets for each time series
        ArrayList<Shapelet> seriesShapelets;
        // construct a map for our K-shapelets lists, on for each classVal.
        if (kShapeletsMap == null) {
            kShapeletsMap = new TreeMap();
            for (int i = 0; i < data.numClasses(); i++) {
                kShapeletsMap.put((double) i, new ArrayList<>());
            }
        }
        // found out how many shapelets we want from each class, split evenly.
        int proportion = numShapelets / kShapeletsMap.keySet().size();

        outputPrint("Processing data for numShapelets " + numShapelets + " with proportion per class = " + proportion);
        outputPrint("in contract balanced: Contract (secs)" + contractTime / 1000000000.0);
        long prevEarlyAbandons = 0;
        int passes = 0;

        // continue processing series until we run out of time (if contracted)
        while (casesSoFar < numSeriesToUse && keepGoing) {
            // outputPrint("BALANCED: "+casesSoFar +" Cumulative time (secs) =
            // "+usedTime/1000000000.0+" Contract time (secs) ="+contractTime/1000000000.0+"
            // contracted = "+contracted+" search type = "+searchFunction.getSearchType());
            // get the Shapelets list based on the classValue of our current time series.
            kShapelets = kShapeletsMap.get(data.get(casesSoFar).classValue());
            // we only want to pass in the worstKShapelet if we've found K shapelets. but we
            // only care about
            // this class values worst one. This is due to the way we represent each classes
            // shapelets in the map.
            worstShapelet = kShapelets.size() == proportion ? kShapelets.get(kShapelets.size() - 1) : null;

            // set the series we're working with.
            shapeletDistance.setSeries(casesSoFar);
            // set the class value of the series we're working with.
            classValue.setShapeletValue(data.get(casesSoFar));
            long t1 = System.nanoTime();
            seriesShapelets = current.searchForShapeletsInSeries(data.get(casesSoFar), this::checkCandidate);
            long t2 = System.nanoTime();
            numShapeletsEvaluated += seriesShapelets.size();

            if (adaptiveTiming && contracted && passes == 0) {
                long tempEA = numEarlyAbandons - prevEarlyAbandons;
                prevEarlyAbandons = numEarlyAbandons;
                double newTimePerShapelet = (double) (t2 - t1) / (seriesShapelets.size() + tempEA);
                if (totalShapeletsPerSeries < (seriesShapelets.size() + tempEA))// Switch to full enum for next
                                                                                // iteration
                    current = full;
                else
                    current = searchFunction;

                System.out.println(" time for case " + casesSoFar + " evaluate  " + seriesShapelets.size()
                        + " but what about early ones?");
                System.out.println(" Est time per shapelet  " + timePerShapelet / 1000000000 + " actual "
                        + newTimePerShapelet / 1000000000);
                shapeletsSearchedPerSeries = adjustNumberPerSeries(contractTime - usedTime, numSeriesToUse - casesSoFar,
                        newTimePerShapelet);
                System.out.println("Changing number of shapelets sampled from "
                        + searchFunction.getNumShapeletsPerSeries() + " to " + shapeletsSearchedPerSeries);
                searchFunction.setNumShapeletsPerSeries(shapeletsSearchedPerSeries);
                System.out.println("data : " + casesSoFar + " has " + seriesShapelets.size() + " candidates"
                        + " cumulative early abandons " + numEarlyAbandons + " worst so far =" + worstShapelet
                        + " evaluated this series = " + (seriesShapelets.size() + tempEA));
            }
            if (seriesShapelets != null) {
                Collections.sort(seriesShapelets, shapeletComparator);
                if (isRemoveSelfSimilar())
                    seriesShapelets = removeSelfSimilar(seriesShapelets);

                kShapelets = combine(proportion, kShapelets, seriesShapelets);
            }

            // re-update the list because it's changed now.
            kShapeletsMap.put(data.get(casesSoFar).classValue(), kShapelets);
            casesSoFar++;
            createSerialFile();
            usedTime = System.nanoTime() - startTime;
            // Logic is we have underestimated the contract so can run back through. If we
            // over estimate it we will just stop.
            if (contracted) {
                if (casesSoFar == numSeriesToUse - 1 && !searchFunction.getSearchType().equals("FULL")) { /// HORRIBLE!
                    casesSoFar = 0;
                    passes++;
                }
                if (usedTime > contractTime)
                    keepGoing = false;
            }
        }

        kShapelets = buildKShapeletsFromMap(kShapeletsMap);

        this.numShapelets = kShapelets.size();

        if (recordShapelets)
            recordShapelets(kShapelets, this.ouputFileLocation);
        // if (!supressOutput)
        // writeShapelets(kShapelets, new OutputStreamWriter(System.out));

        return kShapelets;
    }

    protected ArrayList<Shapelet> buildKShapeletsFromMap(Map<Double, ArrayList<Shapelet>> kShapeletsMap) {
        ArrayList<Shapelet> kShapelets = new ArrayList<>();

        int numberOfClassVals = kShapeletsMap.keySet().size();
        int proportion = numShapelets / numberOfClassVals;

        Iterator<Shapelet> it;

        // all lists should be sorted.
        // go through the map and get the sub portion of best shapelets for the final
        // list.
        for (ArrayList<Shapelet> list : kShapeletsMap.values()) {
            int i = 0;
            it = list.iterator();

            while (it.hasNext() && i++ <= proportion) {
                kShapelets.add(it.next());
            }
        }
        return kShapelets;
    }

    public ArrayList<Shapelet> findBestKShapeletsOriginal(TimeSeriesInstances data) {

        ShapeletSearch full = new ShapeletSearch(searchFunction.getOptions());

        boolean contracted = true;
        boolean keepGoing = true;
        if (contractTime == 0) {
            contracted = false;
        }
        long startTime = System.nanoTime();
        long usedTime = 0;
        int numSeriesToUse = data.numInstances(); // This can be used to reduce the number of series in favour of more

        ArrayList<Shapelet> seriesShapelets; // temp store of all shapelets for each time series
        int dataSize = data.numInstances();
        // for all possible time series.
        long prevEarlyAbandons = 0;
        int passes = 0;
        while (casesSoFar < numSeriesToUse && keepGoing) {
            // outputPrint("ORIGINAL: "+casesSoFar +" Cumulative time (secs) =
            // "+usedTime/1000000000.0+" Contract time (secs) ="+contractTime/1000000000.0+"
            // search type = "+searchFunction.getSearchType());
            // set the worst Shapelet so far, as long as the shapelet set is full.
            worstShapelet = kShapelets.size() == numShapelets ? kShapelets.get(numShapelets - 1) : null;

            // set the series we're working with.
            shapeletDistance.setSeries(casesSoFar);
            // set the class value of the series we're working with.
            classValue.setShapeletValue(data.get(casesSoFar));

            long t1 = System.nanoTime();
            seriesShapelets = searchFunction.searchForShapeletsInSeries(data.get(casesSoFar), this::checkCandidate);
            long t2 = System.nanoTime();
            numShapeletsEvaluated += seriesShapelets.size();

            if (adaptiveTiming && contracted && passes == 0) {
                long tempEA = numEarlyAbandons - prevEarlyAbandons;
                prevEarlyAbandons = numEarlyAbandons;
                double newTimePerShapelet = (double) (t2 - t1) / (seriesShapelets.size() + tempEA);
                outputPrint(" time for case " + casesSoFar + " evaluate  " + seriesShapelets.size()
                        + " but what about early ones?");
                outputPrint(" Est time per shapelet  " + timePerShapelet / 1000000000 + " actual "
                        + newTimePerShapelet / 1000000000);
                shapeletsSearchedPerSeries = adjustNumberPerSeries(contractTime - usedTime, numSeriesToUse - casesSoFar,
                        newTimePerShapelet);
                outputPrint("Changing number of shapelets sampled from " + searchFunction.getNumShapeletsPerSeries()
                        + " to " + shapeletsSearchedPerSeries);
                searchFunction.setNumShapeletsPerSeries(shapeletsSearchedPerSeries);
                outputPrint("data : " + casesSoFar + " has " + seriesShapelets.size() + " candidates"
                        + " cumulative early abandons " + numEarlyAbandons + " worst so far =" + worstShapelet
                        + " evaluated this series = " + (seriesShapelets.size() + tempEA));
            }
            if (seriesShapelets != null) {
                Collections.sort(seriesShapelets, shapeletComparator);

                if (isRemoveSelfSimilar())
                    seriesShapelets = removeSelfSimilar(seriesShapelets);
                kShapelets = combine(numShapelets, kShapelets, seriesShapelets);
            }
            casesSoFar++;
            createSerialFile();
            usedTime = System.nanoTime() - startTime;
            // Logic is we have underestimated the contract so can run back through. If we
            // over estimate it we will just stop.
            if (casesSoFar == numSeriesToUse - 1 && !searchFunction.getSearchType().equals("FULL")) { /// HORRIBLE!
                casesSoFar = 0;
                passes++;
            }
            if (contracted) {
                if (usedTime > contractTime)
                    keepGoing = false;
            }

        }
        this.numShapelets = kShapelets.size();

        if (recordShapelets)
            recordShapelets(kShapelets, this.ouputFileLocation);
        // if (!supressOutput)
        // writeShapelets(kShapelets, new OutputStreamWriter(System.out));

        return kShapelets;
    }

    /*
     * This just goes case by case with no class balancing. With two class problems,
     * we found this better
     */
    public ArrayList<Shapelet> findBestKShapeletsOriginal(Instances data) {

        ShapeletSearch full = new ShapeletSearch(searchFunction.getOptions());

        boolean contracted = true;
        boolean keepGoing = true;
        if (contractTime == 0) {
            contracted = false;
        }
        long startTime = System.nanoTime();
        long usedTime = 0;
        int numSeriesToUse = data.numInstances(); // This can be used to reduce the number of series in favour of more

        ArrayList<Shapelet> seriesShapelets; // temp store of all shapelets for each time series
        int dataSize = data.numInstances();
        // for all possible time series.
        long prevEarlyAbandons = 0;
        int passes = 0;
        while (casesSoFar < numSeriesToUse && keepGoing) {
            // outputPrint("ORIGINAL: "+casesSoFar +" Cumulative time (secs) =
            // "+usedTime/1000000000.0+" Contract time (secs) ="+contractTime/1000000000.0+"
            // search type = "+searchFunction.getSearchType());
            // set the worst Shapelet so far, as long as the shapelet set is full.
            worstShapelet = kShapelets.size() == numShapelets ? kShapelets.get(numShapelets - 1) : null;

            // set the series we're working with.
            shapeletDistance.setSeries(casesSoFar);
            // set the class value of the series we're working with.
            classValue.setShapeletValue(data.get(casesSoFar));

            long t1 = System.nanoTime();
            seriesShapelets = searchFunction.searchForShapeletsInSeries(data.get(casesSoFar), this::checkCandidate);
            long t2 = System.nanoTime();
            numShapeletsEvaluated += seriesShapelets.size();

            if (adaptiveTiming && contracted && passes == 0) {
                long tempEA = numEarlyAbandons - prevEarlyAbandons;
                prevEarlyAbandons = numEarlyAbandons;
                double newTimePerShapelet = (double) (t2 - t1) / (seriesShapelets.size() + tempEA);
                outputPrint(" time for case " + casesSoFar + " evaluate  " + seriesShapelets.size()
                        + " but what about early ones?");
                outputPrint(" Est time per shapelet  " + timePerShapelet / 1000000000 + " actual "
                        + newTimePerShapelet / 1000000000);
                shapeletsSearchedPerSeries = adjustNumberPerSeries(contractTime - usedTime, numSeriesToUse - casesSoFar,
                        newTimePerShapelet);
                outputPrint("Changing number of shapelets sampled from " + searchFunction.getNumShapeletsPerSeries()
                        + " to " + shapeletsSearchedPerSeries);
                searchFunction.setNumShapeletsPerSeries(shapeletsSearchedPerSeries);
                outputPrint("data : " + casesSoFar + " has " + seriesShapelets.size() + " candidates"
                        + " cumulative early abandons " + numEarlyAbandons + " worst so far =" + worstShapelet
                        + " evaluated this series = " + (seriesShapelets.size() + tempEA));
            }
            if (seriesShapelets != null) {
                Collections.sort(seriesShapelets, shapeletComparator);

                if (isRemoveSelfSimilar())
                    seriesShapelets = removeSelfSimilar(seriesShapelets);
                kShapelets = combine(numShapelets, kShapelets, seriesShapelets);
            }
            casesSoFar++;
            createSerialFile();
            usedTime = System.nanoTime() - startTime;
            // Logic is we have underestimated the contract so can run back through. If we
            // over estimate it we will just stop.
            if (casesSoFar == numSeriesToUse - 1 && !searchFunction.getSearchType().equals("FULL")) { /// HORRIBLE!
                casesSoFar = 0;
                passes++;
            }
            if (contracted) {
                if (usedTime > contractTime)
                    keepGoing = false;
            }

        }
        this.numShapelets = kShapelets.size();

        if (recordShapelets)
            recordShapelets(kShapelets, this.ouputFileLocation);
        // if (!supressOutput)
        // writeShapelets(kShapelets, new OutputStreamWriter(System.out));

        return kShapelets;
    }

    private long adjustNumberPerSeries(long timeRemaining, int seriesRemaining, double lastTimePerShapelet) {
        // reinforce time per shapelet
        timePerShapelet = (1 - beta) * timePerShapelet + beta * lastTimePerShapelet;
        // Find time left per series
        long timePerSeries = timeRemaining / seriesRemaining;
        // Find how many we think we can do in that time
        long shapeletsPerSeries = (long) (timePerSeries / timePerShapelet);
        if (shapeletsPerSeries < 1)
            return 1;
        return shapeletsPerSeries;
    }

    public void createSerialFile() {
        if (serialName == null)
            return;

        // Serialise the object.
        ObjectOutputStream out = null;
        try {
            out = new ObjectOutputStream(new FileOutputStream(serialName));
            out.writeObject(this);
        } catch (IOException ex) {
            System.out.println("Failed to write " + ex);
        } finally {
            if (out != null) {
                try {
                    out.close();
                } catch (IOException ex) {
                    System.out.println("Failed to close " + ex);
                }
            }
        }
    }

    /**
     * protected method for extracting k shapelets.
     *
     * @param numShapelets
     * @param data              the data that the shapelets will be taken from
     * @param minShapeletLength
     * @param maxShapeletLength
     * @return an ArrayList of FullShapeletTransform objects in order of their
     *         fitness (by infoGain, seperationGap then shortest length)
     */
    public ArrayList<Shapelet> findBestKShapelets(int numShapelets, Instances data, int minShapeletLength,
            int maxShapeletLength) {
        this.numShapelets = numShapelets;
        // setup classsValue
        classValue.init(data);
        // setup subseqDistance
        shapeletDistance.init(data);
        Instances newData = orderDataSet(data);
        return findBestKShapelets(newData);
    }

    /**
     * Private method to combine two ArrayList collections of FullShapeletTransform
     * objects.
     *
     * @param k                   the maximum number of shapelets to be returned
     *                            after combining the two lists
     * @param kBestSoFar          the (up to) k best shapelets that have been
     *                            observed so far, passed in to combine with
     *                            shapelets from a new series (sorted)
     * @param timeSeriesShapelets the shapelets taken from a new series that are to
     *                            be merged in descending order of fitness with the
     *                            kBestSoFar
     * @return an ordered ArrayList of the best k (or less) (sorted)
     *         FullShapeletTransform objects from the union of the input ArrayLists
     */
    protected ArrayList<Shapelet> combine(int k, ArrayList<Shapelet> kBestSoFar,
            ArrayList<Shapelet> timeSeriesShapelets) {
        // both kBestSofar and timeSeries are sorted so we can exploit this.
        // maintain a pointer for each list.
        ArrayList<Shapelet> newBestSoFar = new ArrayList<>();

        // best so far pointer
        int bsfPtr = 0;
        // new time seris pointer.
        int tssPtr = 0;

        for (int i = 0; i < k; i++) {
            Shapelet shapelet1 = null, shapelet2 = null;

            if (bsfPtr < kBestSoFar.size()) {
                shapelet1 = kBestSoFar.get(bsfPtr);
            }
            if (tssPtr < timeSeriesShapelets.size()) {
                shapelet2 = timeSeriesShapelets.get(tssPtr);
            }

            boolean shapelet1Null = shapelet1 == null;
            boolean shapelet2Null = shapelet2 == null;

            // both lists have been explored, but we have less than K elements.
            if (shapelet1Null && shapelet2Null) {
                break;
            }

            // one list is expired keep adding the other list until we reach K.
            if (shapelet1Null) {
                // even if the list has expired don't just add shapelets without considering
                // they may be dupes.
                AddToBestSoFar(shapelet2, newBestSoFar);
                tssPtr++;
                continue;
            }

            // one list is expired keep adding the other list until we reach K.
            if (shapelet2Null) {
                // even if the list has expired don't just add shapelets without considering
                // they may be dupes.
                AddToBestSoFar(shapelet1, newBestSoFar);
                bsfPtr++;
                continue;
            }

            // if both lists are fine then we need to compare which one to use.
            int compare = shapeletComparator.compare(shapelet1, shapelet2);
            if (compare < 0) {
                AddToBestSoFar(shapelet1, newBestSoFar);
                bsfPtr++;
                shapelet1 = null;
            } else {
                AddToBestSoFar(shapelet2, newBestSoFar);
                tssPtr++;
                shapelet2 = null;

            }
        }

        return newBestSoFar;
    }

    private void AddToBestSoFar(Shapelet shapelet1, ArrayList<Shapelet> newBestSoFar) {
        boolean containsMatchingShapelet = false;
        if (pruneMatchingShapelets)
            containsMatchingShapelet = containsMatchingShapelet(shapelet1, newBestSoFar);

        if (!containsMatchingShapelet)
            newBestSoFar.add(shapelet1);
    }

    private boolean containsMatchingShapelet(Shapelet shapelet, ArrayList<Shapelet> newBestSoFar) {
        // we're going to be comparing all the shapelets we have to shapelet.
        this.shapeletDistance.setShapelet(shapelet);

        // go backwards from where we're at until we stop matching. List is sorted.
        for (int index = newBestSoFar.size() - 1; index >= 0; index--) {
            Shapelet shape = newBestSoFar.get(index);
            int compare2 = shapeletComparator.compare(shape, shapelet);
            // if we are not simply equal to the shapelet that we're looking at then abandon
            // ship.
            if (compare2 != 0) {
                return false; // stop evaluating. we no longer have any matches.
            }

            // if we're here then evaluate the shapelet distance. if they're equal in the
            // comparator it means same length, same IG.
            double dist = this.shapeletDistance.distanceToShapelet(shape);
            // if we hit a shapelet we nearly match with 1e-6 match with stop checking.
            if (NumUtils.isNearlyEqual(dist, 0.0)) {
                return true; // this means we should not add the shapelet.
            }
        }

        return false;
    }

    /**
     * protected method to remove self-similar shapelets from an ArrayList (i.e. if
     * they come from the same series and have overlapping indicies)
     *
     * @param shapelets the input Shapelets to remove self similar
     *                  FullShapeletTransform objects from
     * @return a copy of the input ArrayList with self-similar shapelets removed
     */
    protected static ArrayList<Shapelet> removeSelfSimilar(ArrayList<Shapelet> shapelets) {
        // return a new pruned array list - more efficient than removing
        // self-similar entries on the fly and constantly reindexing
        ArrayList<Shapelet> outputShapelets = new ArrayList<>();
        int size = shapelets.size();
        boolean[] selfSimilar = new boolean[size];

        for (int i = 0; i < size; i++) {
            if (selfSimilar[i]) {
                continue;
            }

            outputShapelets.add(shapelets.get(i));

            for (int j = i + 1; j < size; j++) {
                // no point recalc'ing if already self similar to something
                if ((!selfSimilar[j]) && selfSimilarity(shapelets.get(i), shapelets.get(j))) {
                    selfSimilar[j] = true;
                }
            }
        }
        return outputShapelets;
    }

    protected Shapelet checkCandidate(TimeSeriesInstance series, int start, int length, int dimension) {
        // init qualityBound.
        initQualityBound(classValue.getClassDistributions());

        // Set bound of the bounding algorithm
        if (worstShapelet != null) {
            quality.setBsfQuality(worstShapelet.qualityValue);
        }

        // set the candidate. This is the instance, start and length.
        shapeletDistance.setCandidate(series, start, length, dimension);

        // create orderline by looping through data set and calculating the subsequence
        // distance from candidate to all data, inserting in order.
        ArrayList<OrderLineObj> orderline = new ArrayList<>();

        int dataSize = inputData.numInstances();

        for (int i = 0; i < dataSize; i++) {

            // Check if it is possible to prune the candidate
            if (quality.pruneCandidate()) {
                numEarlyAbandons++;
                return null;
            }

            double distance = 0.0;
            // don't compare the shapelet to the the time series it came from because we
            // know it's 0.
            if (i != casesSoFar) {
                distance = shapeletDistance.calculate(inputDataTS.get(i), i);
            }

            // this could be binarised or normal.
            double classVal = classValue.getClassValue(inputDataTS.get(i));

            // without early abandon, it is faster to just add and sort at the end
            orderline.add(new OrderLineObj(distance, classVal));

            // Update qualityBound - presumably each bounding method for different quality
            // measures will have a different update procedure.
            quality.updateOrderLine(orderline.get(orderline.size() - 1));
        }

        Shapelet shapelet = new Shapelet(shapeletDistance.getCandidate(), dataSourceIDs[casesSoFar], start,
                quality.getQualityMeasure());

        // this class distribution could be binarised or normal.
        shapelet.calculateQuality(orderline, classValue.getClassDistributions());
        shapelet.classValue = classValue.getShapeletValue(); // set classValue of shapelet. (interesing to know).
        shapelet.dimension = dimension;
        return shapelet;
    }

    protected Shapelet checkCandidate(Instance series, int start, int length, int dimension) {
        // init qualityBound.
        initQualityBound(classValue.getClassDistributions());

        // Set bound of the bounding algorithm
        if (worstShapelet != null) {
            quality.setBsfQuality(worstShapelet.qualityValue);
        }

        // set the candidate. This is the instance, start and length.
        shapeletDistance.setCandidate(series, start, length, dimension);

        // create orderline by looping through data set and calculating the subsequence
        // distance from candidate to all data, inserting in order.
        ArrayList<OrderLineObj> orderline = new ArrayList<>();

        int dataSize = inputData.numInstances();

        for (int i = 0; i < dataSize; i++) {

            // Check if it is possible to prune the candidate
            if (quality.pruneCandidate()) {
                numEarlyAbandons++;
                return null;
            }

            double distance = 0.0;
            // don't compare the shapelet to the the time series it came from because we
            // know it's 0.
            if (i != casesSoFar) {
                distance = shapeletDistance.calculate(inputData.instance(i), i);
            }

            // this could be binarised or normal.
            double classVal = classValue.getClassValue(inputData.instance(i));

            // without early abandon, it is faster to just add and sort at the end
            orderline.add(new OrderLineObj(distance, classVal));

            // Update qualityBound - presumably each bounding method for different quality
            // measures will have a different update procedure.
            quality.updateOrderLine(orderline.get(orderline.size() - 1));
        }

        Shapelet shapelet = new Shapelet(shapeletDistance.getCandidate(), dataSourceIDs[casesSoFar], start,
                quality.getQualityMeasure());

        // this class distribution could be binarised or normal.
        shapelet.calculateQuality(orderline, classValue.getClassDistributions());
        shapelet.classValue = classValue.getShapeletValue(); // set classValue of shapelet. (interesing to know).
        shapelet.dimension = dimension;
        return shapelet;
    }

    /**
     * Load a set of Instances from an ARFF
     *
     * @param fileName the file name of the ARFF
     * @return a set of Instances from the ARFF
     */
    public static Instances loadData(String fileName) {
        Instances data = null;
        try {
            FileReader r;
            r = new FileReader(fileName);
            data = new Instances(r);

            data.setClassIndex(data.numAttributes() - 1);
        } catch (IOException e) {
            System.out.println(" Error =" + e + " in method loadData");
        }
        return data;
    }

    /**
     * A private method to assess the self similarity of two FullShapeletTransform
     * objects (i.e. whether they have overlapping indicies and are taken from the
     * same time series).
     *
     * @param shapelet  the first FullShapeletTransform object (in practice, this
     *                  will be the dominant shapelet with quality >= candidate)
     * @param candidate the second FullShapeletTransform
     * @return
     */
    private static boolean selfSimilarity(Shapelet shapelet, Shapelet candidate) {
        // check whether they're the same dimension or not.
        if (candidate.seriesId == shapelet.seriesId && candidate.dimension == shapelet.dimension) {
            if (candidate.startPos >= shapelet.startPos
                    && candidate.startPos < shapelet.startPos + shapelet.getLength()) { // candidate starts within
                                                                                        // exisiting shapelet
                return true;
            }
            if (shapelet.startPos >= candidate.startPos
                    && shapelet.startPos < candidate.startPos + candidate.getLength()) {
                return true;
            }
        }
        return false;
    }

    /**
     * A method to read in a FullShapeletTransform log file to reproduce a
     * FullShapeletTransform
     * <p>
     * NOTE: assumes shapelets from log are Z-NORMALISED
     *
     * @param fileName the name and path of the log file
     * @return a duplicate FullShapeletTransform to the object that created the
     *         original log file
     * @throws Exception
     */
    public static ShapeletTransform createFilterFromFile(String fileName) throws Exception {
        return createFilterFromFile(fileName, Integer.MAX_VALUE);
    }

    /**
     * A method to obtain time taken to find a single best shapelet in the data set
     *
     * @param data              the data set to be processed
     * @param minShapeletLength minimum shapelet length
     * @param maxShapeletLength maximum shapelet length
     * @return time in seconds to find the best shapelet
     */
    public double timingForSingleShapelet(Instances data, int minShapeletLength, int maxShapeletLength) {
        data = roundRobinData(data, null);
        long startTime = System.nanoTime();
        findBestKShapelets(1, data, minShapeletLength, maxShapeletLength);
        long finishTime = System.nanoTime();
        return (double) (finishTime - startTime) / 1000000000.0;
    }

    public void writeAdditionalData(String saveDirectory, int fold) {
        recordShapelets(this.kShapelets, saveDirectory + "_shapelets" + fold + ".csv");
    }

    public void recordShapelets(ArrayList<Shapelet> kShapelets, String saveLocation) {
        // just in case the file doesn't exist or the directories.
        File file = new File(saveLocation);
        if (file.getParentFile() != null) {
            file.getParentFile().mkdirs();
        }

        try (FileWriter out = new FileWriter(file)) {
            writeShapelets(kShapelets, out);
        } catch (IOException ex) {
            Logger.getLogger(ShapeletTransform.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    protected void writeShapelets(ArrayList<Shapelet> kShapelets, OutputStreamWriter out) {
        try {
            out.append("informationGain,seriesId,startPos,classVal,numChannels,dimension\n");
            for (Shapelet kShapelet : kShapelets) {
                out.append(kShapelet.qualityValue + "," + kShapelet.seriesId + "," + kShapelet.startPos + ","
                        + kShapelet.classValue + "," + kShapelet.getNumDimensions() + "," + kShapelet.dimension + "\n");
                for (int i = 0; i < kShapelet.numDimensions; i++) {
                    double[] shapeletContent = kShapelet.getContent().getShapeletContent(i);
                    for (int j = 0; j < shapeletContent.length; j++) {
                        out.append(shapeletContent[j] + ",");
                    }
                    out.append("\n");
                }
            }
        } catch (IOException ex) {
            Logger.getLogger(ShapeletTransform.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    /**
     * Returns a list of the lengths of the shapelets found by this transform.
     *
     * @return An ArrayList of Integers representing the lengths of the shapelets.
     */
    public ArrayList<Integer> getShapeletLengths() {
        ArrayList<Integer> shapeletLengths = new ArrayList<>();

        if (shapelets != null) {
            for (Shapelet s : this.shapelets) {
                shapeletLengths.add(s.getLength());
            }
        }

        return shapeletLengths;
    }

    /**
     * A method to read in a FullShapeletTransform log file to reproduce a
     * FullShapeletTransform,
     * <p>
     * NOTE: assumes shapelets from log are Z-NORMALISED
     *
     * @param fileName     the name and path of the log file
     * @param maxShapelets
     * @return a duplicate FullShapeletTransform to the object that created the
     *         original log file
     * @throws Exception
     */
    public static ShapeletTransform createFilterFromFile(String fileName, int maxShapelets) throws Exception {

        File input = new File(fileName);
        Scanner scan = new Scanner(input);
        scan.useDelimiter("\n");

        ShapeletTransform sf = new ShapeletTransform();
        ArrayList<Shapelet> shapelets = new ArrayList<>();

        String shapeletContentString;
        String shapeletStatsString;
        ArrayList<Double> content;
        double[] contentArray;
        Scanner lineScan;
        Scanner statScan;
        double qualVal;
        int serID;
        int starPos;

        int shapeletCount = 0;

        while (shapeletCount < maxShapelets && scan.hasNext()) {
            shapeletStatsString = scan.next();
            shapeletContentString = scan.next();

            // Get the shapelet stats
            statScan = new Scanner(shapeletStatsString);
            statScan.useDelimiter(",");

            qualVal = Double.parseDouble(statScan.next().trim());
            serID = Integer.parseInt(statScan.next().trim());
            starPos = Integer.parseInt(statScan.next().trim());
            // End of shapelet stats

            lineScan = new Scanner(shapeletContentString);
            lineScan.useDelimiter(",");

            content = new ArrayList<>();
            while (lineScan.hasNext()) {
                String next = lineScan.next().trim();
                if (!next.isEmpty()) {
                    content.add(Double.parseDouble(next));
                }
            }

            contentArray = new double[content.size()];
            for (int i = 0; i < content.size(); i++) {
                contentArray[i] = content.get(i);
            }

            contentArray = sf.shapeletDistance.seriesRescaler.rescaleSeries(contentArray, false);

            ShapeletCandidate cand = new ShapeletCandidate();
            cand.setShapeletContent(contentArray);

            Shapelet s = new Shapelet(cand, qualVal, serID, starPos);

            shapelets.add(s);
            shapeletCount++;
        }
        sf.shapelets = shapelets;
        sf.searchComplete = true;
        sf.numShapelets = shapelets.size();
        sf.setShapeletMinAndMax(1, 1);

        return sf;
    }

    /**
     * A method to read in a shapelet csv file and return a shapelet arraylist.
     * 
     * @param f
     * @return a duplicate FullShapeletTransform to the object that created the
     *         original log file
     * @throws FileNotFoundException
     */
    public static ArrayList<Shapelet> readShapeletCSV(File f) throws FileNotFoundException {
        ArrayList<Shapelet> shapelets = new ArrayList<>();

        Scanner sc = new Scanner(f);
        System.out.println(sc.nextLine());

        boolean readHeader = true;

        double quality = 0, classVal = 0;
        int series = 0, position = 0, dimension = 0, numDimensions = 1;
        ShapeletCandidate cand = null;
        int currentDim = 0;

        while (sc.hasNextLine()) {
            String line = sc.nextLine();
            String[] cotentsAsString = line.split(",");

            if (readHeader) {
                quality = Double.parseDouble(cotentsAsString[0]);
                series = Integer.parseInt(cotentsAsString[1]);
                position = Integer.parseInt(cotentsAsString[2]);
                classVal = Double.parseDouble(cotentsAsString[3]);
                numDimensions = Integer.parseInt(cotentsAsString[4]);
                dimension = Integer.parseInt(cotentsAsString[5]);
                cand = new ShapeletCandidate(numDimensions);
                currentDim = 0;
                readHeader = false;
            } else {
                // read dims until we run out.
                double[] content = new double[cotentsAsString.length];
                for (int i = 0; i < content.length; i++) {
                    content[i] = Double.parseDouble(cotentsAsString[i]);
                }
                // set the content for the current channel.
                cand.setShapeletContent(currentDim, content);
                currentDim++;

                // if we've evald all the current dim data for a shapelet we can add it to the
                // list, and move on with the next one.
                if (currentDim == numDimensions) {
                    Shapelet shapelet = new Shapelet(cand, quality, series, position);
                    shapelet.dimension = dimension;
                    shapelet.classValue = classVal;
                    shapelets.add(shapelet);
                    readHeader = true;
                }
            }
        }
        return shapelets;

    }

    /**
     * Method to reorder the given Instances in round robin order
     *
     * @param data      Instances to be reordered
     * @param sourcePos Pointer to array of ints, where old positions of instances
     *                  are to be stored.
     * @return Instances in round robin order
     */
    public static Instances roundRobinData(Instances data, int[] sourcePos) {
        // Count number of classes
        TreeMap<Double, ArrayList<Instance>> instancesByClass = new TreeMap<>();
        TreeMap<Double, ArrayList<Integer>> positionsByClass = new TreeMap<>();

        NormalClassValue ncv = new NormalClassValue();
        ncv.init(data);

        // Get class distributions
        ClassCounts classDistribution = ncv.getClassDistributions();

        // Allocate arrays for instances of every class
        for (int i = 0; i < classDistribution.size(); i++) {
            int frequency = classDistribution.get(i);
            instancesByClass.put((double) i, new ArrayList<>(frequency));
            positionsByClass.put((double) i, new ArrayList<>(frequency));
        }

        int dataSize = data.numInstances();
        // Split data according to their class memebership
        for (int i = 0; i < dataSize; i++) {
            Instance inst = data.instance(i);
            instancesByClass.get(ncv.getClassValue(inst)).add(inst);
            positionsByClass.get(ncv.getClassValue(inst)).add(i);
        }

        // Merge data into single list in round robin order
        Instances roundRobinData = new Instances(data, dataSize);
        for (int i = 0; i < dataSize;) {
            // Allocate arrays for instances of every class
            for (int j = 0; j < classDistribution.size(); j++) {
                ArrayList<Instance> currentList = instancesByClass.get((double) j);
                ArrayList<Integer> currentPositions = positionsByClass.get((double) j);

                if (!currentList.isEmpty()) {
                    roundRobinData.add(currentList.remove(currentList.size() - 1));
                    if (sourcePos != null && sourcePos.length == dataSize) {
                        sourcePos[i] = currentPositions.remove(currentPositions.size() - 1);
                    }
                    i++;
                }
            }
        }

        return roundRobinData;
    }

        /**
     * Method to reorder the given Instances in round robin order
     *
     * @param data      Instances to be reordered
     * @param sourcePos Pointer to array of ints, where old positions of instances
     *                  are to be stored.
     * @return Instances in round robin order
     */
    public static TimeSeriesInstances roundRobinData(TimeSeriesInstances data, int[] sourcePos) {
        // Count number of classes
        TreeMap<Double, ArrayList<TimeSeriesInstance>> instancesByClass = new TreeMap<>();
        TreeMap<Double, ArrayList<Integer>> positionsByClass = new TreeMap<>();

        NormalClassValue ncv = new NormalClassValue();
        ncv.init(data);

        // Get class distributions
        ClassCounts classDistribution = ncv.getClassDistributions();

        // Allocate arrays for instances of every class
        for (int i = 0; i < classDistribution.size(); i++) {
            int frequency = classDistribution.get(i);
            instancesByClass.put((double) i, new ArrayList<>(frequency));
            positionsByClass.put((double) i, new ArrayList<>(frequency));
        }

        int dataSize = data.numInstances();
        // Split data according to their class memebership
        for (int i = 0; i < dataSize; i++) {
            TimeSeriesInstance inst = data.get(i);
            instancesByClass.get(ncv.getClassValue(inst)).add(inst);
            positionsByClass.get(ncv.getClassValue(inst)).add(i);
        }

        // Merge data into single list in round robin order
        TimeSeriesInstances roundRobinData = new TimeSeriesInstances(data.getClassLabels());
        for (int i = 0; i < dataSize;) {
            // Allocate arrays for instances of every class
            for (int j = 0; j < classDistribution.size(); j++) {
                ArrayList<TimeSeriesInstance> currentList = instancesByClass.get((double) j);
                ArrayList<Integer> currentPositions = positionsByClass.get((double) j);

                if (!currentList.isEmpty()) {
                    roundRobinData.add(currentList.remove(currentList.size() - 1));
                    if (sourcePos != null && sourcePos.length == dataSize) {
                        sourcePos[i] = currentPositions.remove(currentPositions.size() - 1);
                    }
                    i++;
                }
            }
        }

        return roundRobinData;
    }

    public void outputPrint(String val) {
        if (!this.supressOutput) {
            System.out.println(val);
        }
    }

    @Override
    public String toString() {
        String str = "Shapelets: \n";
        for (Shapelet s : shapelets) {
            str += s.toString() + "\n";
        }
        return str;
    }

    public String getShapeletCounts() {
        return "numShapelets," + numShapelets + ",numShapeletsEvaluated," + numShapeletsEvaluated + ",numEarlyAbandons,"
                + numEarlyAbandons;
    }

    // searchFunction

    public String getParameters() {
        String str = "minShapeletLength," + searchFunction.getMin() + ",maxShapeletLength," + searchFunction.getMax()
                + ",numShapelets," + numShapelets + ",numShapeletsEvaluated," + numShapeletsEvaluated
                + ",numEarlyAbandons," + numEarlyAbandons + ",searchFunction," + this.searchFunction.getSearchType()
                + ",qualityMeasure," + this.quality.getQualityMeasure().getClass().getSimpleName() + ",subseqDistance,"
                + this.shapeletDistance.getClass().getSimpleName() + ",roundrobin," + roundRobin + ",earlyAbandon,"
                + useCandidatePruning + ",TransformClass," + this.getClass().getSimpleName();
        return str;
    }

    /**
     *
     * @param data
     * @param minShapeletLength
     * @param maxShapeletLength
     * @return
     * @throws Exception
     */
    public long opCountForSingleShapelet(Instances data, int minShapeletLength, int maxShapeletLength)
            throws Exception {
        data = roundRobinData(data, null);
        subseqDistOpCount = 0;
        findBestKShapelets(1, data, minShapeletLength, maxShapeletLength);
        return subseqDistOpCount;
    }

    public static void main(String[] args) {

    }

    /**
     * @return the removeSelfSimilar
     */
    public boolean isRemoveSelfSimilar() {
        return removeSelfSimilar;
    }

    /**
     * @param removeSelfSimilar the removeSelfSimilar to set
     */
    public void setRemoveSelfSimilar(boolean removeSelfSimilar) {
        this.removeSelfSimilar = removeSelfSimilar;
    }

    /**
     * @return the pruneMatchingShapelets
     */
    public boolean isPruneMatchingShapelets() {
        return pruneMatchingShapelets;
    }

    /**
     * @param pruneMatchingShapelets the pruneMatchingShapelets to set
     */
    public void setPruneMatchingShapelets(boolean pruneMatchingShapelets) {
        this.pruneMatchingShapelets = pruneMatchingShapelets;
    }

    public void setClassValue(NormalClassValue cv) {
        classValue = cv;
    }

    public void setSearchFunction(ShapeletSearch shapeletSearch) {
        searchFunction = shapeletSearch;
    }

    public ShapeletSearch getSearchFunction() {
        return searchFunction;
    }

    public void setSerialName(String sName) {
        serialName = sName;
    }

    public void useSeparationGap() {
        shapeletComparator = new Shapelet.ReverseSeparationGap();
    }

    public void setShapeletComparator(Comparator<Shapelet> comp) {
        shapeletComparator = comp;
    }

    public void setUseRoundRobin(boolean b) {
        useRoundRobin = b;
    }

    public ShapeletDistance getSubSequenceDistance() {
        return shapeletDistance;
    }

    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;
        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "authors");
        result.setValue(TechnicalInformation.Field.YEAR, "put in Aarons paper");
        result.setValue(TechnicalInformation.Field.TITLE, "stuff");
        result.setValue(TechnicalInformation.Field.JOURNAL, "places");
        result.setValue(TechnicalInformation.Field.VOLUME, "vol");
        result.setValue(TechnicalInformation.Field.PAGES, "pages");

        return result;
    }

    public void turnOffLog() {
        this.recordShapelets = false;
    }

    public void supressOutput() {
        this.supressOutput = true;
    }

    /**
     * Use candidate pruning technique when checking candidate quality. This speeds
     * up the transform processing time.
     */
    public void useCandidatePruning() {
        this.useCandidatePruning = true;
        this.candidatePruningStartPercentage = 10;
    }

    /**
     * Use candidate pruning technique when checking candidate quality. This speeds
     * up the transform processing time.
     *
     * @param percentage the percentage of data to be preprocessed before pruning is
     *                   initiated. In most cases the higher the percentage the less
     *                   effective pruning becomes
     */
    public void useCandidatePruning(int percentage) {
        this.useCandidatePruning = true;
        this.candidatePruningStartPercentage = percentage;
    }

    /********************** SETTERS ************************************/

    /**
     * Shouldnt really have this method, but it is a convenience to allow
     * refactoring ClusteredShapeletTransform
     * 
     * @param s
     */
    public void setShapelets(ArrayList<Shapelet> s) {
        this.shapelets = s;
    }

    /**
     * Set the transform to round robin the data or not. This transform defaults
     * round robin to false to keep the instances in the same order as the original
     * data. If round robin is set to true, the transformed data will be reordered
     * which can make it more difficult to use the ensemble.
     *
     * @param val
     */
    public void setRoundRobin(boolean val) {
        this.roundRobin = val;
    }

    public void setUseBalancedClasses(boolean val) {
        this.useBalancedClasses = val;
    }

    public void setSuppressOutput(boolean b) {
        this.supressOutput = !b;
    }

    public void setNumberOfShapelets(int k) {
        this.numShapelets = k;
    }

    /**
     * Set file path for the filter log. Filter log includes shapelet quality,
     * seriesId, startPosition, and content for each shapelet.
     *
     * @param fileName the updated file path of the filter log
     */
    public void setLogOutputFile(String fileName) {
        this.recordShapelets = true;
        this.ouputFileLocation = fileName;
    }

    /**
     * Mutator method to set the minimum and maximum shapelet lengths for the
     * filter.
     *
     * @param min minimum length of shapelets
     * @param max maximum length of shapelets
     */
    public void setShapeletMinAndMax(int min, int max) {
        searchFunction.setMinAndMax(min, max);
    }

    public void setQualityMeasure(ShapeletQualityChoice qualityChoice) {
        quality = new ShapeletQuality(qualityChoice);
    }

    public void setRescaler(SeriesRescaler rescaler) {
        if (shapeletDistance != null)
            this.shapeletDistance.seriesRescaler = rescaler;
    }

    public void setCandidatePruning(boolean f) {
        this.useCandidatePruning = f;
        this.candidatePruningStartPercentage = f ? 10 : 100;
    }

    public void setContractTime(long c) {
        contractTime = c;
    }

    public void setAdaptiveTiming(boolean b) {
        adaptiveTiming = b;
    }

    public void setTimePerShapelet(double t) {
        timePerShapelet = t;
    }

    public void setShapeletDistance(ShapeletDistance ssd) {
        shapeletDistance = ssd;
    }

    /*************** GETTERS *************/
    public long getCount() {
        return count;
    }

    public ShapeletQualityChoice getQualityMeasure() {
        return quality.getChoice();
    }

    public int getNumberOfShapelets() {
        return numShapelets;
    }

    public long getNumShapeletsPerSeries() {
        return searchFunction.getNumShapeletsPerSeries();
    }

    public ArrayList<Shapelet> getShapelets() {
        return this.shapelets;
    }

    public boolean getSuppressOutput() {
        return this.supressOutput;
    }

    @Override
    public boolean isFit() {
        return searchComplete;
    }



}
