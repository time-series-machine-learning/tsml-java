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
package tsml.filters.shapelet_filters;

import experiments.data.DatasetLoading;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.OutputStreamWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Scanner;
import java.util.TreeMap;
import java.util.logging.Level;
import java.util.logging.Logger;

import tsml.transformers.shapelet_tools.*;
import tsml.transformers.shapelet_tools.distance_functions.ShapeletDistance;
import utilities.ClassifierTools;
import utilities.class_counts.ClassCounts;
import weka.classifiers.meta.RotationForest;
import weka.core.*;
import weka.filters.SimpleBatchFilter;
import tsml.transformers.shapelet_tools.class_value.BinaryClassValue;
import tsml.transformers.shapelet_tools.class_value.NormalClassValue;
import tsml.transformers.shapelet_tools.quality_measures.ShapeletQuality;
import tsml.transformers.shapelet_tools.quality_measures.ShapeletQuality.ShapeletQualityChoice;
import tsml.transformers.shapelet_tools.search_functions.ShapeletSearch;
import tsml.transformers.shapelet_tools.search_functions.ShapeletSearchOptions;
import tsml.transformers.shapelet_tools.distance_functions.ImprovedOnlineShapeletDistance;
import tsml.transformers.shapelet_tools.search_functions.ShapeletSearchFactory;
import utilities.rescalers.SeriesRescaler;

/**

 * NOTE: As shapelet extraction can be time consuming, there is an option to output shapelets
 * to a text file (Default location is in the root dir of the project, file name "defaultShapeletOutput.txt").
 *
 * Default settings are TO NOT PRODUCE OUTPUT FILE - unless file name is changed, each successive filter will
 * overwrite the output (see "setLogOutputFile(String fileName)" to change file dir and name).
 *
 * To reconstruct a filter from this output, please see the method "createFilterFromFile(String fileName)".
 * 
 * 
 * 
 * A filter to transform a dataset by k shapelets. Once built on a training set,
 * the filter can be used to transform subsequent datasets using the extracted
 * shapelets.
 * <p>
 * See <a
 * href="http://delivery.acm.org/10.1145/2340000/2339579/p289-lines.pdf?ip=139.222.14.198&acc=ACTIVE%20SERVICE&CFID=221649628&CFTOKEN=31860141&__acm__=1354814450_3dacfa9c5af84445ea2bfd7cc48180c8">
 * Lines J., Davis, L., Hills, J., Bagnall, A.: A shapelet transform for time series
 * classification. In: Proc. 18th ACM SIGKDD (2012)</a>
 *
 * @author Aaron Bostrom
 */
public class ShapeletFilter extends SimpleBatchFilter implements Serializable,TechnicalInformationHandler{
//Global defaults. Max should be a lambda set to series length
    public final static int DEFAULT_NUMSHAPELETS = 500;
    public final static int DEFAULT_MINSHAPELETLENGTH = 3;
    public final static int DEFAULT_MAXSHAPELETLENGTH = 23;

    //Variables for experiments
    protected static long subseqDistOpCount;
    private boolean removeSelfSimilar = true;
    private boolean pruneMatchingShapelets;

    //this int is used to serialise our position when iterating through a dataset.
    public int casesSoFar;
    public boolean searchComplete=false;
    
    protected boolean supressOutput=true; // defaults to print in System.out AS WELL as file, set to true to stop printing to console
    protected int numShapelets; //The maximum number of shapelets in the transform. This is K and is different to the total number of shapelets to look for/looked for

    protected ArrayList<Shapelet> shapelets;
    protected String ouputFileLocation = "defaultShapeletOutput.txt"; // default store location
    protected boolean recordShapelets; // default action is to write an output file
    protected boolean roundRobin;
    protected long numShapeletsEvaluated=0;//This counts the total number of shapelets returned by searchForShapeletsInSeries. It does not include early abandoned shapelets
    protected long numEarlyAbandons=0;//This counts number of shapelets early abandoned
    protected transient ShapeletQuality quality;
    
    /*protected transient QualityMeasures.ShapeletQualityMeasure qualityMeasure;
    protected transient QualityMeasures.ShapeletQualityChoice qualityChoice;
    protected transient QualityBound.ShapeletQualityBound qualityBound;*/
    
    protected boolean useCandidatePruning;
    protected boolean useRoundRobin;
    protected boolean useBalancedClasses;

    protected Comparator<Shapelet> shapeletComparator;
//Made public for now to keep the filters
    public ShapeletDistance subseqDistance;

    public NormalClassValue classValue;

    protected ShapeletSearch searchFunction;
    protected String serialName;
    protected Shapelet worstShapelet;
    
    protected Instances inputData;
    
    protected ArrayList<Shapelet> kShapelets;
    
    protected long count;


    public void setSubSeqDistance(ShapeletDistance ssd) {
        subseqDistance = ssd;
    }
    
    public long getCount() {
        return count;
    }



    protected int candidatePruningStartPercentage;

    protected static final double ROUNDING_ERROR_CORRECTION = 0.000000000000001;
    protected int[] dataSourceIDs;

    /**
     * Default constructor; Quality measure defaults to information gain.
     */
    public ShapeletFilter() {
        this(DEFAULT_NUMSHAPELETS, DEFAULT_MINSHAPELETLENGTH, DEFAULT_MAXSHAPELETLENGTH, ShapeletQualityChoice.INFORMATION_GAIN);
    }
    
    /**
     * Constructor for generating a shapelet transform from an ArrayList of
     * Shapelets.
     *
     * @param shapes
     */
    public ShapeletFilter(ArrayList<Shapelet> shapes) {
        this();
        this.shapelets = shapes;
        this.m_FirstBatchDone = true;
        this.numShapelets = shapelets.size();
    }

    /**
     * Single param constructor: Quality measure defaults to information gain.
     *
     * @param k the number of shapelets to be generated
     */
    public ShapeletFilter(int k) {
        this(k, DEFAULT_MINSHAPELETLENGTH, DEFAULT_MAXSHAPELETLENGTH, ShapeletQualityChoice.INFORMATION_GAIN);
    }

    /**
     * Full constructor to create a usable filter. Quality measure defaults to
     * information gain.
     *
     * @param k the number of shapelets to be generated
     * @param minShapeletLength minimum length of shapelets
     * @param maxShapeletLength maximum length of shapelets
     */
    public ShapeletFilter(int k, int minShapeletLength, int maxShapeletLength) {
        this(k, minShapeletLength, maxShapeletLength, ShapeletQualityChoice.INFORMATION_GAIN);

    }

    /**
     * Full, exhaustive, constructor for a filter. Quality measure set via enum,
     * invalid selection defaults to information gain.
     *
     * @param k the number of shapelets to be generated
     * @param minShapeletLength minimum length of shapelets
     * @param maxShapeletLength maximum length of shapelets
     * @param qualityChoice the shapelet quality measure to be used with this
     * filter
     */    
    public ShapeletFilter(int k, int minShapeletLength, int maxShapeletLength, ShapeletQualityChoice qualityChoice) {
        this.numShapelets = k;
        this.shapelets = new ArrayList<>();
        this.m_FirstBatchDone = false;
        this.useCandidatePruning = false;
        this.casesSoFar = 0;
        this.recordShapelets = true; // default action is to write an output file
        this.roundRobin = false;
        this.useRoundRobin = false;
        this.shapeletComparator = new Shapelet.LongOrder();
        this.kShapelets = new ArrayList<>();

        setQualityMeasure(qualityChoice);
        this.subseqDistance = new ShapeletDistance();
        this.classValue = new NormalClassValue();
        
        ShapeletSearchOptions sOp = new ShapeletSearchOptions.Builder().setMin(minShapeletLength).setMax(maxShapeletLength).build();   
        this.searchFunction = new ShapeletSearchFactory(sOp).getShapeletSearch();
    }
    public long getNumShapeletsPerSeries(){ return searchFunction.getNumShapeletsPerSeries();}
    /**
     * Returns the set of shapelets for this transform as an ArrayList.
     *
     * @return An ArrayList of Shapelets representing the shapelets found for
     * this Shapelet Transform.
     */
    public ArrayList<Shapelet> getShapelets() {
        return this.shapelets;
    }

    /**
     * Shouldnt really hav this method, but it is a convenience to allow refactoring
     * ClusteredShapeletTransform
     * @param s
     */
    public void setShapelets(ArrayList<Shapelet> s) {
        this.shapelets=s;
    }


    /**
     * Set the transform to round robin the data or not. This transform defaults
     * round robin to false to keep the instances in the same order as the
     * original data. If round robin is set to true, the transformed data will
     * be reordered which can make it more difficult to use the ensemble.
     *
     * @param val
     */
    public void setRoundRobin(boolean val) {
        this.roundRobin = val;
    }
    public void setUseBalancedClasses(boolean val) {
        this.useBalancedClasses = val;
    }

    /**
     * Supresses filter output to the console; useful when running timing
     * experiments.
     */
    public void supressOutput() {
        this.supressOutput = true;
    }


    public void setSuppressOutput(boolean b) {
        this.supressOutput = !b;
    }
    public boolean getSuppressOutput() {
        return this.supressOutput;
    }


    /**
     * Use candidate pruning technique when checking candidate quality. This
     * speeds up the transform processing time.
     */
    public void useCandidatePruning() {
        this.useCandidatePruning = true;
        this.candidatePruningStartPercentage = 10;
    }

    /**
     * Use candidate pruning technique when checking candidate quality. This
     * speeds up the transform processing time.
     *
     * @param percentage the percentage of data to be precocessed before pruning
     * is initiated. In most cases the higher the percentage the less effective
     * pruning becomes
     */
    public void useCandidatePruning(int percentage) {
        this.useCandidatePruning = true;
        this.candidatePruningStartPercentage = percentage;
    }

    /**
     * Mutator method to set the number of shapelets to be stored by the filter.
     *
     * @param k the number of shapelets to be generated
     */
    public void setNumberOfShapelets(int k) {
        this.numShapelets = k;
    }

    /**
     * set the number of shapelets in the transform
     * @return
     */
    public int getNumberOfShapelets() {
        return numShapelets;
    }

    /**
     * Turns off log saving; useful for timing experiments where speed is
     * essential.
     */
    public void turnOffLog() {
        this.recordShapelets = false;
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

    /**
     * Mutator method to set the quality measure used by the filter. As with
     * constructors, default selection is information gain unless another valid
     * selection is specified.
     *
     * @return
     */
    public ShapeletQualityChoice getQualityMeasure() {
        return quality.getChoice();
    }

    /**
     *
     * @param qualityChoice
     */
    public void setQualityMeasure(ShapeletQualityChoice qualityChoice) {
        quality = new ShapeletQuality(qualityChoice);
    }
    
    
     /**
     *
     * @param rescaler
     */
    public void setRescaler(SeriesRescaler rescaler){
        if(subseqDistance != null)
            this.subseqDistance.seriesRescaler = rescaler;
    }
    
        /**
     *
     * @param classDist
     * @return
     */
    protected void initQualityBound(ClassCounts classDist) {
        if (!useCandidatePruning) return;
        quality.initQualityBound(classDist, candidatePruningStartPercentage);
    }

    /**
     *
     * @param f
     */
    public void setCandidatePruning(boolean f) {
        this.useCandidatePruning = f;
        this.candidatePruningStartPercentage = f ? 10 : 100;
    }

    /**
     * Sets the format of the filtered instances that are output. I.e. will
     * include k attributes each shapelet distance and a class value
     *
     * @param inputFormat the format of the input data
     * @return a new Instances object in the desired output format
     */
    @Override
    protected Instances determineOutputFormat(Instances inputFormat) throws IllegalArgumentException {

        if (this.numShapelets < 1) {
            
            System.out.println(this.numShapelets);
            throw new IllegalArgumentException("ShapeletTransform not initialised correctly - please specify a value of k (this.numShapelets) that is greater than or equal to 1. It is currently set tp "+this.numShapelets);
        }

        //Set up instances size and format.
        //int length = this.numShapelets;
        int length = this.shapelets.size();
        ArrayList<Attribute> atts = new ArrayList<>();
        String name;
        for (int i = 0; i < length; i++) {
            name = "Shapelet_" + i;
            atts.add(new Attribute(name));
        }

        if (inputFormat.classIndex() >= 0) {
            //Classification set, set class
            //Get the class values as a fast vector
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

    protected void inputCheck(Instances dataInst) throws IllegalArgumentException {
        if (numShapelets < 1) {
            throw new IllegalArgumentException("Number of shapelets initialised incorrectly to "+numShapelets+" - please select value of k (Usage: setNumberOfShapelets");
        }

        int maxPossibleLength;
        maxPossibleLength = dataInst.instance(0).numAttributes();

        if (dataInst.classIndex() >= 0) {
            maxPossibleLength -= 1;
        }
    }

    /**
     * The main logic of the filter; when called for the first time, k shapelets
     * are extracted from the input Instances 'data'. The input 'data' is
     * transformed by the k shapelets, and the filtered data is returned as an
     * output.
     * <p>
     * If called multiple times, shapelet extraction DOES NOT take place again;
     * once k shapelets are established from the initial call to process(), the
     * k shapelets are used to transform subsequent Instances.
     * <p>
     * Intended use:
     * <p>
     * 1. Extract k shapelets from raw training data to build filter;
     * <p>
     * 2. Use the filter to transform the raw training data into transformed
     * training data;
     * <p>
     * 3. Use the filter to transform the raw testing data into transformed
     * testing data (e.g. filter never extracts shapelets from training data,
     * therefore avoiding bias);
     * <p>
     * 4. Build a classifier using transformed training data, perform
     * classification on transformed test data.
     *
     * @param data the input data to be transformed (and to find the shapelets
     * if this is the first run)
     * @return the transformed representation of data, according to the
     * distances from each instance to each of the k shapelets
     */
    @Override
    public Instances process(Instances data) throws IllegalArgumentException {
        inputData = data;
        //check the input data is correct and assess whether the filter has been setup correctly.
        inputCheck(data);

        //checks if the shapelets haven't been found yet, finds them if it needs too.
        if (!m_FirstBatchDone && !searchComplete){
            trainShapelets(data);
            searchComplete=true;
            //we log the count from the subsequence distance before we reset it in the transform.
            //we only care about the count from the train. What is it counting?
            count = subseqDistance.getCount();

        }

        //build the transformed dataset with the shapelets we've found either on this data, or the previous training data
        return buildTansformedDataset(data);
    }

    protected void trainShapelets(Instances data) {
        //we might round robin the data in here. So we need to override the input data with the new ordering.
        inputData = initDataSource(data);
        searchFunction.setComparator(shapeletComparator);
        searchFunction.init(inputData);
                //setup subseqDistance
        subseqDistance.init(inputData);
        //setup classValue
        classValue.init(inputData);
        outputPrint("num shapelets before search "+numShapelets);

        shapelets = findBestKShapeletsCache(inputData); // get k shapelets
        m_FirstBatchDone = true;
        outputPrint(shapelets.size() + " Shapelets have been generated num shapelets now "+numShapelets);
        
        //we don't need to undo the roundRobin because we clone the data into a different order.
    }
    
    private Instances initDataSource(Instances data) {

        int dataSize = data.numInstances();
        // shapelets discovery has not yet been carried out, so this must be training data
        dataSourceIDs = new int[dataSize];
        
        Instances dataset = data;
        if (roundRobin) {
            //Reorder the data in round robin order
            dataset = roundRobinData(data, dataSourceIDs);
        } else {
            for (int i = 0; i < dataSize; i++) {
                dataSourceIDs[i] = i;
            }
        }
        
        return dataset;
    }

    //given a set of instances transform it by the internal shapelets.
    public Instances buildTansformedDataset(Instances data) {
        //Reorder the training data and reset the shapelet indexes
        Instances output = determineOutputFormat(data);

        //init out data for transforming.
        subseqDistance.init(inputData);
        //setup classsValue
        classValue.init(inputData);

        Shapelet s;
        // for each data, get distance to each shapelet and create new instance
        int size = shapelets.size();
        int dataSize = data.numInstances();

        //create our data instances
        for (int j = 0; j < dataSize; j++) {
            output.add(new DenseInstance(size + 1));
        }

        double dist;
        for (int i = 0; i < size; i++) {
            s = shapelets.get(i);
            subseqDistance.setShapelet(s);

            for (int j = 0; j < dataSize; j++) {
                dist = subseqDistance.calculate(data.instance(j), j);
                output.instance(j).setValue(i, dist);
            }
        }

        //do the classValues.
        for (int j = 0; j < dataSize; j++) {
            //we always want to write the true ClassValue here. Irrelevant of binarised or not.
            output.instance(j).setValue(size, data.instance(j).classValue());
        }

        return output;
    }

    /**
     * protected method for extracting k shapelets.
     * this method extracts shapelets series by series, using the searchFunction method searchForShapeletsInSeries,
     *  which itself uses checkCandidate
     * 1. The search method determines the method of choosing shapelets. By default all are evaluated (ShapeletSearch)
     * or the alternative RandomSearch, which finds a fixed number of shapelets determined by the time contract.
     * 2. The qualityFunction assesses each candidate, and uses the worstShapelet (set in checkCandidate) to test for inclusion and
     * any lower bounding. I dont think it uses it to test for inclusion.
     * 3. self similar are removed by default, and the method combine is used to merge the current candidates and
     * the new ones
     * @param data the data that the shapelets will be taken from
     * @return an ArrayList of FullShapeletTransform objects in order of their
     * fitness (by infoGain, seperationGap then shortest length)
     */
    public ArrayList<Shapelet> findBestKShapeletsCache(Instances data) {
        ArrayList<Shapelet> seriesShapelets;                                    // temp store of all shapelets for each time series
        // temp store of all shapelets for each time series
        //for all time series
        outputPrint("Processing data: ");

        int dataSize = data.numInstances();
        //for all possible time series.
        for(; casesSoFar < dataSize; casesSoFar++) {

            //set the worst Shapelet so far, as long as the shapelet set is full.
            worstShapelet = kShapelets.size() == numShapelets ? kShapelets.get(numShapelets - 1) : null;

            //set the series we're working with.
            subseqDistance.setSeries(casesSoFar);
            //set the class value of the series we're working with.
            classValue.setShapeletValue(data.get(casesSoFar));

            seriesShapelets = searchFunction.searchForShapeletsInSeries(data.get(casesSoFar), this::checkCandidate);
            numShapeletsEvaluated+=seriesShapelets.size();
            outputPrint("data : " + casesSoFar+" has "+seriesShapelets.size()+" candidates"+ " cumulative early abandons "+numEarlyAbandons+" worst so far ="+worstShapelet);
            if(seriesShapelets != null){
                Collections.sort(seriesShapelets, shapeletComparator);
                
                if(isRemoveSelfSimilar())
                    seriesShapelets = removeSelfSimilar(seriesShapelets);
                kShapelets = combine(numShapelets, kShapelets, seriesShapelets);
            }
            
            createSerialFile();
        }

        this.numShapelets = kShapelets.size();

        if (recordShapelets) 
            recordShapelets(kShapelets, this.ouputFileLocation);
        if (!supressOutput)
            writeShapelets(kShapelets, new OutputStreamWriter(System.out));

        return kShapelets;
    }
    
    public void createSerialFile()
    {
        if(serialName == null) return;
        
        //Serialise the object.
        ObjectOutputStream out = null;
        try {
            out = new ObjectOutputStream(new FileOutputStream(serialName));
            out.writeObject(this);
        } catch (IOException ex) {
            System.out.println("Failed to write " + ex);
        }
        finally{
            if(out != null){
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
     * @param data the data that the shapelets will be taken from
     * @param minShapeletLength
     * @param maxShapeletLength
     * @return an ArrayList of FullShapeletTransform objects in order of their
     * fitness (by infoGain, seperationGap then shortest length)
     */
    public ArrayList<Shapelet> findBestKShapeletsCache(int numShapelets, Instances data, int minShapeletLength, int maxShapeletLength) {
        this.numShapelets = numShapelets;
        //setup classsValue
        classValue.init(data);
        //setup subseqDistance
        subseqDistance.init(data);
        Instances newData=initDataSource(data);
        return findBestKShapeletsCache(newData);
    }



    /**
     * Private method to combine two ArrayList collections of
     * FullShapeletTransform objects.
     *
     * @param k the maximum number of shapelets to be returned after combining
     * the two lists
     * @param kBestSoFar the (up to) k best shapelets that have been observed so
     * far, passed in to combine with shapelets from a new series (sorted)
     * @param timeSeriesShapelets the shapelets taken from a new series that are
     * to be merged in descending order of fitness with the kBestSoFar
     * @return an ordered ArrayList of the best k (or less) (sorted)
     * FullShapeletTransform objects from the union of the input ArrayLists
     */
    protected ArrayList<Shapelet> combine(int k, ArrayList<Shapelet> kBestSoFar, ArrayList<Shapelet> timeSeriesShapelets) {
        //both kBestSofar and timeSeries are sorted so we can exploit this.
        //maintain a pointer for each list.
        ArrayList<Shapelet> newBestSoFar = new ArrayList<>();

        //best so far pointer
        int bsfPtr = 0;
        //new time seris pointer.
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

            //both lists have been explored, but we have less than K elements.
            if (shapelet1Null && shapelet2Null) {
                break;
            }

            //one list is expired keep adding the other list until we reach K.
            if (shapelet1Null) {
                //even if the list has expired don't just add shapelets without considering they may be dupes.
                AddToBestSoFar(shapelet2, newBestSoFar);
                tssPtr++;
                continue;
            }

            //one list is expired keep adding the other list until we reach K.
            if (shapelet2Null) {
                //even if the list has expired don't just add shapelets without considering they may be dupes.
                AddToBestSoFar(shapelet1, newBestSoFar);
                bsfPtr++;
                continue;
            }

            //if both lists are fine then we need to compare which one to use.
            int compare = shapeletComparator.compare(shapelet1, shapelet2);
            if (compare < 0) {
                AddToBestSoFar(shapelet1, newBestSoFar);
                bsfPtr++;
                shapelet1 = null;
            } else{
                AddToBestSoFar(shapelet2, newBestSoFar);
                tssPtr++;
                shapelet2 = null;
            
            }             
        }

        return newBestSoFar;
    }

    private void AddToBestSoFar(Shapelet shapelet1, ArrayList<Shapelet> newBestSoFar) {
        boolean containsMatchingShapelet = false;
        if(pruneMatchingShapelets)
            containsMatchingShapelet = containsMatchingShapelet(shapelet1, newBestSoFar);
        
        if(!containsMatchingShapelet)
            newBestSoFar.add(shapelet1);
    }

    private boolean containsMatchingShapelet(Shapelet shapelet, ArrayList<Shapelet> newBestSoFar){
        //we're going to be comparing all the shapelets we have to shapelet.
        this.subseqDistance.setShapelet(shapelet);

        //go backwards from where we're at until we stop matching. List is sorted.
        for(int index=newBestSoFar.size()-1;index>=0; index--){
            Shapelet shape = newBestSoFar.get(index);
            int compare2 = shapeletComparator.compare(shape, shapelet);
            //if we are not simply equal to the shapelet that we're looking at then abandon ship.
            if(compare2 != 0){
                return false; // stop evaluating. we no longer have any matches.
            }
            

            //if we're here then evaluate the shapelet distance. if they're equal in the comparator it means same length, same IG.
            double dist = this.subseqDistance.distanceToShapelet(shape);
            //if we hit a shapelet we nearly match with 1e-6 match with stop checking.  
            if(isNearlyEqual(dist, 0.0)){
                return true; //this means we should not add the shapelet.
            }
        }
                    
        return false;    
    }
    
    private static boolean isNearlyEqual(double a, double b){
        double eps = 1e-6;
        return Math.abs(a - b) < eps;
    }
    

    /**
     * protected method to remove self-similar shapelets from an ArrayList (i.e.
     * if they come from the same series and have overlapping indicies)
     *
     * @param shapelets the input Shapelets to remove self similar
     * FullShapeletTransform objects from
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

    protected Shapelet checkCandidate(Instance series, int start, int length, int dimension) {
        //init qualityBound.        
        initQualityBound(classValue.getClassDistributions());        
        
        //Set bound of the bounding algorithm
        if (worstShapelet != null) {
            quality.setBsfQuality(worstShapelet.qualityValue);
        }
        
        //set the candidate. This is the instance, start and length.
        subseqDistance.setCandidate(series, start, length, dimension);

        // create orderline by looping through data set and calculating the subsequence
        // distance from candidate to all data, inserting in order.
        ArrayList<OrderLineObj> orderline = new ArrayList<>();

        int dataSize = inputData.numInstances();

        for (int i = 0; i < dataSize; i++) {
            
            //Check if it is possible to prune the candidate
            if (quality.pruneCandidate()) {
                numEarlyAbandons++;
                return null;
            }

            double distance = 0.0;
            //don't compare the shapelet to the the time series it came from because we know it's 0.
            if (i != casesSoFar) {
                distance = subseqDistance.calculate(inputData.instance(i), i);
            }

            //this could be binarised or normal. 
            double classVal = classValue.getClassValue(inputData.instance(i));

            // without early abandon, it is faster to just add and sort at the end
            orderline.add(new OrderLineObj(distance, classVal));

            //Update qualityBound - presumably each bounding method for different quality measures will have a different update procedure.
            quality.updateOrderLine(orderline.get(orderline.size() - 1));
        }

        Shapelet shapelet = new Shapelet(subseqDistance.getCandidate(), dataSourceIDs[casesSoFar], start, quality.getQualityMeasure());
        
        //this class distribution could be binarised or normal.
        shapelet.calculateQuality(orderline, classValue.getClassDistributions());
        shapelet.classValue = classValue.getShapeletValue(); //set classValue of shapelet. (interesing to know).
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
     * A private method to assess the self similarity of two
     * FullShapeletTransform objects (i.e. whether they have overlapping
     * indicies and are taken from the same time series).
     *
     * @param shapelet the first FullShapeletTransform object (in practice, this
     * will be the dominant shapelet with quality >= candidate)
     * @param candidate the second FullShapeletTransform
     * @return
     */
    private static boolean selfSimilarity(Shapelet shapelet, Shapelet candidate) {
        //check whether they're the same dimension or not.
        if (candidate.seriesId == shapelet.seriesId && candidate.dimension == shapelet.dimension) {
            if (candidate.startPos >= shapelet.startPos && candidate.startPos < shapelet.startPos + shapelet.getLength()) { //candidate starts within exisiting shapelet
                return true;
            }
            if (shapelet.startPos >= candidate.startPos && shapelet.startPos < candidate.startPos + candidate.getLength()) {
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
     * original log file
     * @throws Exception
     */
    public static ShapeletFilter createFilterFromFile(String fileName) throws Exception {
        return createFilterFromFile(fileName, Integer.MAX_VALUE);
    }

        /**
     * A method to obtain time taken to find a single best shapelet in the data
     * set
     *
     * @param data the data set to be processed
     * @param minShapeletLength minimum shapelet length
     * @param maxShapeletLength maximum shapelet length
     * @return time in seconds to find the best shapelet
     */
    public double timingForSingleShapelet(Instances data, int minShapeletLength, int maxShapeletLength) {
        data = roundRobinData(data, null);
        long startTime = System.nanoTime();
        findBestKShapeletsCache(1, data, minShapeletLength, maxShapeletLength);
        long finishTime = System.nanoTime();
        return (double) (finishTime - startTime) / 1000000000.0;
    }

    
    public void writeAdditionalData(String saveDirectory, int fold){
        recordShapelets(this.kShapelets, saveDirectory + "_shapelets" + fold + ".csv");
    }
    
    
    public void recordShapelets(ArrayList<Shapelet> kShapelets, String saveLocation) {
        //just in case the file doesn't exist or the directories.
        File file = new File(saveLocation);
        if (file.getParentFile() != null) {
            file.getParentFile().mkdirs();
        }

        try (FileWriter out = new FileWriter(file)) {
            writeShapelets(kShapelets, out);
        } catch (IOException ex) {
            Logger.getLogger(ShapeletFilter.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    
    protected void writeShapelets(ArrayList<Shapelet> kShapelets, OutputStreamWriter out){
        try {
            out.append("informationGain,seriesId,startPos,classVal,numChannels,dimension\n");
            for (Shapelet kShapelet : kShapelets) {
                out.append(kShapelet.qualityValue + "," + kShapelet.seriesId + "," + kShapelet.startPos + "," + kShapelet.classValue + "," + kShapelet.getNumDimensions() + "," + kShapelet.dimension+"\n");
                for (int i = 0; i < kShapelet.numDimensions; i++) {
                    double[] shapeletContent = kShapelet.getContent().getShapeletContent(i);
                    for (int j = 0; j < shapeletContent.length; j++) {
                        out.append(shapeletContent[j] + ",");
                    }
                    out.append("\n");
                }
            }
        } catch (IOException ex) {
            Logger.getLogger(ShapeletFilter.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    /**
     * Returns a list of the lengths of the shapelets found by this transform.
     *
     * @return An ArrayList of Integers representing the lengths of the
     * shapelets.
     */
    public ArrayList<Integer> getShapeletLengths() {
        ArrayList<Integer> shapeletLengths = new ArrayList<>();

        if (m_FirstBatchDone) {
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
     * @param fileName the name and path of the log file
     * @param maxShapelets
     * @return a duplicate FullShapeletTransform to the object that created the
     * original log file
     * @throws Exception
     */
    public static ShapeletFilter createFilterFromFile(String fileName, int maxShapelets) throws Exception {

        File input = new File(fileName);
        Scanner scan = new Scanner(input);
        scan.useDelimiter("\n");

        ShapeletFilter sf = new ShapeletFilter();
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

            //Get the shapelet stats
            statScan = new Scanner(shapeletStatsString);
            statScan.useDelimiter(",");

            qualVal = Double.parseDouble(statScan.next().trim());
            serID = Integer.parseInt(statScan.next().trim());
            starPos = Integer.parseInt(statScan.next().trim());
            //End of shapelet stats

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

            contentArray = sf.subseqDistance.seriesRescaler.rescaleSeries(contentArray, false);
            
            ShapeletCandidate cand = new ShapeletCandidate();
            cand.setShapeletContent(contentArray);

            Shapelet s = new Shapelet(cand, qualVal, serID, starPos);

            shapelets.add(s);
            shapeletCount++;
        }
        sf.shapelets = shapelets;
        sf.m_FirstBatchDone = true;
        sf.numShapelets = shapelets.size();
        sf.setShapeletMinAndMax(1, 1);

        return sf;
    }

    
    
     /**
     * A method to read in a shapelet csv file and return a shapelet arraylist.

     * @param f
     * @return a duplicate FullShapeletTransform to the object that created the
     * original log file
     * @throws java.io.FileNotFoundException
     */
    public static ArrayList<Shapelet> readShapeletCSV(File f) throws FileNotFoundException{
        ArrayList<Shapelet> shapelets = new ArrayList<>();
        
        Scanner sc = new Scanner(f);
        System.out.println(sc.nextLine());
        
        boolean readHeader = true;
        
        double quality = 0, classVal = 0;
        int series = 0, position = 0, dimension = 0, numDimensions = 1;
        ShapeletCandidate cand = null;
        int currentDim = 0;

        
        while(sc.hasNextLine()){
            String line = sc.nextLine();
            String[] cotentsAsString = line.split(",");

            if(readHeader){
                quality = Double.parseDouble(cotentsAsString[0]);
                series = Integer.parseInt(cotentsAsString[1]);
                position = Integer.parseInt(cotentsAsString[2]);
                classVal = Double.parseDouble(cotentsAsString[3]);
                numDimensions = Integer.parseInt(cotentsAsString[4]);
                dimension = Integer.parseInt(cotentsAsString[5]);
                cand = new ShapeletCandidate(numDimensions);
                currentDim =0;
                readHeader = false;
            }
            else{
                //read dims until we run out.
                double[] content = new double[cotentsAsString.length];
                for (int i = 0; i < content.length; i++) {
                    content[i] = Double.parseDouble(cotentsAsString[i]);
                }
                //set the content for the current channel.
                cand.setShapeletContent(currentDim, content);
                currentDim++;
                
                //if we've evald all the current dim data for a shapelet we can add it to the list, and move on with the next one.
                if(currentDim == numDimensions){
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
     * @param data Instances to be reordered
     * @param sourcePos Pointer to array of ints, where old positions of
     * instances are to be stored.
     * @return Instances in round robin order
     */
    public static Instances roundRobinData(Instances data, int[] sourcePos) {
        //Count number of classes 
        TreeMap<Double, ArrayList<Instance>> instancesByClass = new TreeMap<>();
        TreeMap<Double, ArrayList<Integer>> positionsByClass = new TreeMap<>();

        NormalClassValue ncv = new NormalClassValue();
        ncv.init(data);

        //Get class distributions 
        ClassCounts classDistribution = ncv.getClassDistributions();

        //Allocate arrays for instances of every class
        for (int i = 0; i < classDistribution.size(); i++) {
            int frequency = classDistribution.get(i);
            instancesByClass.put((double) i, new ArrayList<>(frequency));
            positionsByClass.put((double) i, new ArrayList<>(frequency));
        }

        int dataSize = data.numInstances();
        //Split data according to their class memebership
        for (int i = 0; i < dataSize; i++) {
            Instance inst = data.instance(i);
            instancesByClass.get(ncv.getClassValue(inst)).add(inst);
            positionsByClass.get(ncv.getClassValue(inst)).add(i);
        }

        //Merge data into single list in round robin order
        Instances roundRobinData = new Instances(data, dataSize);
        for (int i = 0; i < dataSize;) {
            //Allocate arrays for instances of every class
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
        return "numShapelets," + numShapelets + ",numShapeletsEvaluated," + numShapeletsEvaluated + ",numEarlyAbandons," + numEarlyAbandons;
    }

    //searchFunction

    public String getParameters(){
        String str="minShapeletLength,"+searchFunction.getMin()+",maxShapeletLength,"+searchFunction.getMax()+",numShapelets,"+numShapelets
                +",numShapeletsEvaluated,"+numShapeletsEvaluated+",numEarlyAbandons,"+numEarlyAbandons
                + ",searchFunction,"+this.searchFunction.getSearchType()
                + ",qualityMeasure,"+this.quality.getQualityMeasure().getClass().getSimpleName()
                +",subseqDistance,"+this.subseqDistance.getClass().getSimpleName()
                +",roundrobin,"+roundRobin+",earlyAbandon,"+useCandidatePruning+",TransformClass,"+this.getClass().getSimpleName();
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
    public long opCountForSingleShapelet(Instances data, int minShapeletLength, int maxShapeletLength) throws Exception {
        data = roundRobinData(data, null);
        subseqDistOpCount = 0;
        findBestKShapeletsCache(1, data, minShapeletLength, maxShapeletLength);
        return subseqDistOpCount;
    }
    public static void basicTest(){
        String dataLocation = "E:\\Data\\TSCProblems2018\\";
        String saveLocation = "C:\\Temp\\TSC\\";
        final String dataset = "FordA";
        final int fold = 1;
        final String filePath = dataLocation + File.separator + dataset + File.separator + dataset;
        Instances test, train;
        test = DatasetLoading.loadDataNullable(filePath + "_TEST");
        train = DatasetLoading.loadDataNullable(filePath + "_TRAIN");
         ShapeletSearchOptions searchOptions = new ShapeletSearchOptions.Builder()
                                            .setMin(3)
                                            .setMax(train.numAttributes()-1)
                                            .setSearchType(ShapeletSearch.SearchType.FULL)
                                            .build();


        ShapeletTransformFactoryOptions options = new ShapeletTransformFactoryOptions.ShapeletTransformOptions()
                                            .setDistanceType(ShapeletDistance.DistanceType.IMPROVED_ONLINE)
                                            .setKShapelets(train.numInstances()*10)
                                            .useBinaryClassValue()
                                            .useClassBalancing()
                                            .useCandidatePruning()
                                            .useRoundRobin()
                                            .setSearchOptions(searchOptions)
                                            .build();

        ShapeletFilter transform = new ShapeletTransformFactory(options).getFilter();
        transform.setLogOutputFile(saveLocation+"fordAOutput.csv");

        long startTime1 = System.nanoTime();

        Instances tranTrain1 = transform.process(train);
        Instances tranTest1 = transform.process(test);

        long endTime1 = System.nanoTime();

        System.out.println("Transform time = " + (endTime1-startTime1)); 

    }
    
    public static void testFilterUsage(){
        String filePath ="Z:\\ArchiveData\\Univariate_ts\\";
        String problem="Chinatown";
        Instances test, train;
        test = DatasetLoading.loadDataNullable(filePath + problem+"\\"+problem+"_TEST");
        train = DatasetLoading.loadDataNullable(filePath + problem+"\\"+problem+"_TRAIN");
        ShapeletFilter shapeletFilter = ShapeletTransformTimingUtilities.createTransformWithTimeLimit(train, 24);
        
        
        
    }
    public static void main(String[] args){
        try {
            final String resampleLocation = "D:\\Research TSC\\Data\\TSCProblems2018";
            final String dataset = "Yoga";
            final int fold = 1;
            final String filePath = resampleLocation + File.separator + dataset + File.separator + dataset;
            Instances test, train;
            test = DatasetLoading.loadDataNullable(filePath + "_TEST");
            train = DatasetLoading.loadDataNullable(filePath + "_TRAIN");
            //use fold as the seed.
            //train = InstanceTools.subSample(train, 100, fold);
            
            
            ShapeletFilter transform = new ShapeletFilter();
            transform.setRoundRobin(true);
            //construct shapelet classifiers.
            transform.setClassValue(new BinaryClassValue());
            transform.setSubSeqDistance(new ImprovedOnlineShapeletDistance());
            transform.setShapeletMinAndMax(3, train.numAttributes() - 1);
            transform.useCandidatePruning();
            transform.setNumberOfShapelets(train.numInstances() * 10);
            transform.setQualityMeasure(ShapeletQualityChoice.INFORMATION_GAIN);
            transform.supressOutput();
            transform.setPruneMatchingShapelets(true);
            
            long startTime = System.nanoTime();
           
            Instances tranTrain = transform.process(train);
            Instances tranTest = transform.process(test);
            
            long endTime = System.nanoTime();
            
            RotationForest rot1 = new RotationForest();
            rot1.buildClassifier(tranTrain);
            double accuracy = ClassifierTools.accuracy(tranTest, rot1);
            
            System.out.println("Shapelet transform "+ accuracy + " time " + (endTime-startTime));
            

            ShapeletSearchOptions searchOptions = new ShapeletSearchOptions.Builder()
                                                .setMin(3)
                                                .setMax(train.numAttributes()-1)
                                                .setSearchType(ShapeletSearch.SearchType.FULL)
                                                .build();
                                                
            
            ShapeletTransformFactoryOptions options = new ShapeletTransformFactoryOptions.ShapeletTransformOptions()
                                                .setDistanceType(ShapeletDistance.DistanceType.IMPROVED_ONLINE)
                                                .setKShapelets(train.numInstances()*10)
                                                .useBinaryClassValue()
                                                .useClassBalancing()
                                                .useCandidatePruning()
                                                .useRoundRobin()
                                                .setSearchOptions(searchOptions)
                                                .build();
            
            ShapeletFilter transform1 = new ShapeletTransformFactory(options).getFilter();
            transform1.supressOutput();
            
            long startTime1 = System.nanoTime();
            
            Instances tranTrain1 = transform.process(train);
            Instances tranTest1 = transform.process(test);
            
            long endTime1 = System.nanoTime();
            
            
            RotationForest rot2 = new RotationForest();
            rot2.buildClassifier(tranTrain1);
            double accuracy1 = ClassifierTools.accuracy(tranTest1, rot2);
            
            System.out.println("Fast shapelet transform "+ accuracy1 + " time " + (endTime1-startTime1));
        } catch (Exception ex) {
            Logger.getLogger(ShapeletFilter.class.getName()).log(Level.SEVERE, null, ex);
        }
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
    public ShapeletSearch getSearchFunction(){
        return searchFunction;
    }
    public void setSerialName(String sName) {
        serialName = sName;
    }
    public void useSeparationGap() {
        shapeletComparator = new Shapelet.ReverseSeparationGap();
    }
    public void setShapeletComparator(Comparator<Shapelet> comp){
        shapeletComparator = comp;
    }
    public void setUseRoundRobin(boolean b) {
        useRoundRobin = b;
    }
    public ShapeletDistance getSubSequenceDistance(){
        return subseqDistance;
    }
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation    result;
        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "authors");
        result.setValue(TechnicalInformation.Field.YEAR, "put in Aarons paper");
        result.setValue(TechnicalInformation.Field.TITLE, "stuff");
        result.setValue(TechnicalInformation.Field.JOURNAL, "places");
        result.setValue(TechnicalInformation.Field.VOLUME, "vol");
        result.setValue(TechnicalInformation.Field.PAGES, "pages");

        return result;
    }
    @Override
    public String globalInfo() {
        return "Shapelet Transform with settings =";
    }

}
