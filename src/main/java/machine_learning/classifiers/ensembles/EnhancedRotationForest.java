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

/*
 *    Restructure of ContractRotationForest.java to making bagging easier. An adaptation of Rotation Forest, 2008 Juan Jose Rodriguez
 *      Contract Version by @author Tony Bagnall, Michael Flynn, first implemented 2018, updated 2019 (checkpointable)
 *      and 2020 (conform to structure)
 *
 *
 * We have cloned the code from RotationForest rather than extend it because core changes occur in most methods, and
 * to decouple from Weka, which has removed random forest from the latest releases.
 *
 */


package machine_learning.classifiers.ensembles;

import evaluation.evaluators.CrossValidationEvaluator;
import tsml.classifiers.Checkpointable;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.TrainTimeContractable;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.PrincipalComponents;
import weka.filters.unsupervised.attribute.RemoveUseless;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.File;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.TimeUnit;


public class EnhancedRotationForest extends EnhancedAbstractClassifier
        implements TrainTimeContractable, Checkpointable, Serializable{

    Classifier baseClassifier;
    ArrayList<Classifier> classifiers;
    /** for serialization */
    static final long serialVersionUID = -3255631880798499936L;
    /** The minimum size of a group */
    protected int minGroup = 3;
    /** The maximum size of a group */
    protected int maxGroup = 3;
    /** The percentage of instances to be removed */
    protected int removedPercentage = 50;
    /** The attributes of each group */
    ArrayList< int[][]> groups;
    /** The type of projection filter */
    protected Filter projectionFilter;
    /** The projection filters */
    protected ArrayList<Filter []> projectionFilters;
    /** Headers of the transformed dataset */
    protected ArrayList<Instances> headers;
    /** Headers of the reduced datasets */
    protected ArrayList<Instances []> reducedHeaders;
    /** Filter that remove useless attributes */
    protected RemoveUseless removeUseless = null;
    /** Filter that normalized the attributes */
    protected Normalize normalize = null;


    private boolean trainTimeContract = false;
    transient private long trainContractTimeNanos =0;
    //Added features
    private double estSingleTree;
    //Stores the actual number of trees after the build, may vary with contract
    private int numTrees=0;
    private int minNumTrees=200;
    private int maxNumTrees=200;
    int maxNumAttributes;
    String checkpointPath=null;
    boolean checkpoint=false;
    double timeUsed;

    /** Flags and data required if Bagging **/
    private boolean bagging = false;
    private int[] oobCounts;
    private double[][] trainDistributions;
    /** data information **/
    private int seriesLength;
    private int numInstances;



    /**
     * Constructor.
     */
    public EnhancedRotationForest() {
        super(CAN_ESTIMATE_OWN_PERFORMANCE);

        baseClassifier = new weka.classifiers.trees.J48();
        projectionFilter = defaultFilter();
        checkpointPath=null;
        timeUsed=0;

    }

    /**
     * Default projection method.
     */
    protected Filter defaultFilter() {
        PrincipalComponents filter = new PrincipalComponents();
        //filter.setNormalize(false);
        filter.setVarianceCovered(1.0);
        return filter;
    }


    /**
     * Sets the minimum size of a group.
     *
     * @param minGroup the minimum value.
     * of attributes.
     */
    public void setMinGroup( int minGroup ) throws IllegalArgumentException {

        if( minGroup <= 0 )
            throw new IllegalArgumentException( "MinGroup has to be positive." );
        this.minGroup = minGroup;
    }

    /**
     * Gets the minimum size of a group.
     *
     * @return 		the minimum value.
     */
    public int getMinGroup() {
        return minGroup;
    }

    public void setMaxNumTrees(int t) throws IllegalArgumentException {
        if( t <= 0 )
            throw new IllegalArgumentException( "maxNumTrees has to be positive." );
        maxNumTrees=t;
    }
    public void setMinNumTrees(int t) throws IllegalArgumentException {
        if( t <= 0 )
            throw new IllegalArgumentException( "minNumTrees has to be positive." );
        minNumTrees=t;
    }

    /**
     * Sets the maximum size of a group.
     *
     * @param maxGroup the maximum value.
     * of attributes.
     */
    public void setMaxGroup( int maxGroup ) throws IllegalArgumentException {

        if( maxGroup <= 0 )
            throw new IllegalArgumentException( "MaxGroup has to be positive." );
        this.maxGroup = maxGroup;
    }

    /**
     * Gets the maximum size of a group.
     *
     * @return 		the maximum value.
     */
    public int getMaxGroup() {
        return maxGroup;
    }



    /**
     * Sets the percentage of instance to be removed
     *
     * @param removedPercentage the percentage.
     */
    public void setRemovedPercentage( int removedPercentage ) throws IllegalArgumentException {

        if( removedPercentage < 0 )
            throw new IllegalArgumentException( "RemovedPercentage has to be >=0." );
        if( removedPercentage >= 100 )
            throw new IllegalArgumentException( "RemovedPercentage has to be <100." );

        this.removedPercentage = removedPercentage;
    }

    /**
     * Gets the percentage of instances to be removed
     *
     * @return 		the percentage.
     */
    public int getRemovedPercentage() {
        return removedPercentage;
    }


    /**
     * Sets the filter used to project the data.
     *
     * @param projectionFilter the filter.
     */
    public void setProjectionFilter( Filter projectionFilter ) {

        this.projectionFilter = projectionFilter;
    }

    /**
     * Gets the filter used to project the data.
     *
     * @return 		the filter.
     */
    public Filter getProjectionFilter() {
        return projectionFilter;
    }

    /**
     * Gets the filter specification string, which contains the class name of
     * the filter and any options to the filter
     *
     * @return the filter string.
     */
    /* Taken from FilteredClassifier */
    protected String getProjectionFilterSpec() {

        Filter c = getProjectionFilter();
        if (c instanceof OptionHandler) {
            return c.getClass().getName() + " "
                    + Utils.joinOptions(((OptionHandler)c).getOptions());
        }
        return c.getClass().getName();
    }

    @Override
    public String toString() {
        return "toString not implemented for ContractRotationForest";
    }

    /**
     * builds the classifier.
     *
     * @param data 	the training data to be used for generating the
     * 			classifier.
     * @throws Exception 	if the classifier could not be built successfully
     */
    @Override
    public void buildClassifier(Instances data) throws Exception {
        // can classifier handle the data? These default capabilities
        // only allow real valued series and classification. To be adjusted
        getCapabilities().testWithFail(data);
        long startTime=System.nanoTime();
        //Set up the results file
        super.buildClassifier(data);
//This is from the RotationForest: remove zero variance and normalise attributes.
//Do this before loading from file, so we can perform checks of dataset?
        removeUseless = new RemoveUseless();
        removeUseless.setInputFormat(data);
        data = Filter.useFilter(data, removeUseless);
        normalize = new Normalize();
        normalize.setInputFormat(data);
        data = Filter.useFilter(data, normalize);
        seriesLength = data.numAttributes()-1;
        numInstances = data.numInstances();

        File file = new File(checkpointPath + "RotF" + seed + ".ser");
        //if checkpointing and serialised files exist load said files
        if (checkpoint && file.exists()){ //Configure from file
            printLineDebug("Loading from checkpoint file");
            loadFromFile(checkpointPath + "RotF" + seed + ".ser");
            //               checkpointTimeElapsed -= System.nanoTime()-t1;
        }
        else{   //Initialise:
            groups=new ArrayList<>();
            // These arrays keep the information of the transformed data set
            headers =new ArrayList<>();
            //Store the PCA transforms
            projectionFilters =new ArrayList<>();
            reducedHeaders = new ArrayList<>();
            classifiers=new ArrayList<>();
            numTrees = 0;
            normalize = new Normalize();
            normalize.setInputFormat(data);
            data = Filter.useFilter(data, normalize);

        }
        //Set up for Bagging if required
        if(bagging && getEstimateOwnPerformance()) {
            trainDistributions = new double[numInstances][numClasses];
            oobCounts = new int[numInstances];
        }
        //Can do this just once if not bagging
        Instances [] instancesOfClass;
        instancesOfClass = new Instances[numClasses];
        for( int i = 0; i < instancesOfClass.length; i++ ) {
            instancesOfClass[i] = new Instances( data, 0 );
        }
        do{//Always build at least one tree
//Formed bag data set
            Instances trainData=data;
            boolean[] inBag=null;
            if(bagging){
                //Resample data with replacement
                long t1 = System.nanoTime();
                inBag = new boolean[trainData.numInstances()];
                trainData = trainData.resampleWithWeights(rand, inBag);
            }
//Build classifier
            Classifier c= buildTree(trainData,instancesOfClass,numTrees, data.numAttributes()-1);
            classifiers.add(c);
            if(bagging) { // Get bagged distributions
                for(int i=0;i<data.numInstances();i++){
                    if(!inBag[i]){
                        oobCounts[i]++;
                        double[] dist=c.distributionForInstance(data.instance(i));
                        for(int j=0;j<dist.length;j++)
                            trainDistributions[i][j]+=dist[j];
                    }
                }
            }
//If the first one takes too long, adjust length parameter
            numTrees++;
        }while(withinTrainContract(trainResults.getBuildTime()) && classifiers.size() < minNumTrees);
        //Build the classifier
        trainResults.setBuildTime(System.nanoTime()-startTime);
        trainResults.setParas(getParameters());
        if (getEstimateOwnPerformance()) {
            long est1 = System.nanoTime();
            estimateOwnPerformance(data);
            long est2 = System.nanoTime();

            if (bagging)
                trainResults.setErrorEstimateTime(est2 - est1 + trainResults.getErrorEstimateTime());
            else
                trainResults.setErrorEstimateTime(est2 - est1);

            trainResults.setBuildPlusEstimateTime(trainResults.getBuildTime() + trainResults.getErrorEstimateTime());
        }

        trainResults.setParas(getParameters());
        printLineDebug("*************** Finished RotF Build with " + numTrees + " Trees built in " + (System.nanoTime() - startTime) / 1000000000 + " Seconds  ***************");

    }

    /** Build a rotation forest tree, possibly not using all the attributes to speed things up
     *
     * @param data
     * @param instancesOfClass
     * @param i
     * @param numAtts
     * @throws Exception
     */
    public Classifier buildTree(Instances data, Instances [] instancesOfClass, int i, int numAtts) throws Exception{
        int[][] g=generateGroupFromSize(data, rand,numAtts);
        Filter[] projection=Filter.makeCopies(projectionFilter, g.length );
        projectionFilters.add(projection);
        groups.add(g);
        Instances[] reducedHeaders = new Instances[ g.length ];
        this.reducedHeaders.add(reducedHeaders);

        ArrayList<Attribute> transformedAttributes = new ArrayList<>( data.numAttributes() );
        // Construction of the dataset for each group of attributes
        for( int j = 0; j < g.length; j++ ) {
            ArrayList<Attribute> fv = new ArrayList<>( g[j].length + 1 );
            for( int k = 0; k < g[j].length; k++ ) {
                String newName = data.attribute( g[j][k] ).name()
                        + "_" + k;
                fv.add(data.attribute( g[j][k] ).copy(newName) );
            }
            fv.add( (Attribute)data.classAttribute( ).copy() );
            Instances dataSubSet = new Instances( "rotated-" + i + "-" + j + "-",
                    fv, 0);
            dataSubSet.setClassIndex( dataSubSet.numAttributes() - 1 );
            // Select instances for the dataset
            reducedHeaders[j] = new Instances( dataSubSet, 0 );
            boolean [] selectedClasses = selectClasses( instancesOfClass.length,
                    rand );
            for( int c = 0; c < selectedClasses.length; c++ ) {
                if( !selectedClasses[c] )
                    continue;
                for(Instance instance:instancesOfClass[c]) {
                    Instance newInstance = new DenseInstance(dataSubSet.numAttributes());
                    newInstance.setDataset( dataSubSet );
                    for( int k = 0; k < g[j].length; k++ ) {
                        newInstance.setValue( k, instance.value( g[j][k] ) );
                    }
                    newInstance.setClassValue( instance.classValue( ) );
                    dataSubSet.add( newInstance );
                }
            }
            dataSubSet.randomize(rand);
            // Remove a percentage of the instances
            Instances originalDataSubSet = dataSubSet;
            dataSubSet.randomize(rand);
            RemovePercentage rp = new RemovePercentage();
            rp.setPercentage(removedPercentage );
            rp.setInputFormat( dataSubSet );
            dataSubSet = Filter.useFilter( dataSubSet, rp );
            if( dataSubSet.numInstances() < 2 ) {
                dataSubSet = originalDataSubSet;
            }
            // Project the data

            projection[j].setInputFormat( dataSubSet );
            Instances projectedData = null;
            do {
                try {
                    projectedData = Filter.useFilter( dataSubSet,
                            projection[j] );
                } catch ( Exception e ) {
                    // The data could not be projected, we add some random instances
                    addRandomInstances( dataSubSet, 10, rand );
                }
            } while( projectedData == null );

            // Include the projected attributes in the attributes of the 
            // transformed dataset
            for( int a = 0; a < projectedData.numAttributes() - 1; a++ ) {
                String newName = projectedData.attribute(a).name() + "_" + j;
                transformedAttributes.add( projectedData.attribute(a).copy(newName));
            }
        }

        transformedAttributes.add((Attribute)data.classAttribute().copy() );
        Instances buildClas = new Instances( "rotated-" + i + "-",
                transformedAttributes, 0 );
        buildClas.setClassIndex( buildClas.numAttributes() - 1 );
        headers.add(new Instances( buildClas, 0 ));

        // Project all the training data
        for(Instance instance:data) {
            Instance newInstance = convertInstance( instance, i );
            buildClas.add( newInstance );
        }
        Classifier c= AbstractClassifier.makeCopy(baseClassifier);
        // Build the base classifier
        if (c instanceof Randomizable) {
            ((Randomizable) c).setSeed(rand.nextInt());
        }
        c.buildClassifier( buildClas );
        return c;
    }



    private void estimateOwnPerformance(Instances data) throws Exception {
        if (bagging) {
            // Use bag data, counts normalised to probabilities
            printLineDebug("Finding the OOB estimates");
            double[] preds = new double[data.numInstances()];
            double[] actuals = new double[data.numInstances()];
            long[] predTimes = new long[data.numInstances()];//Dummy variable, need something
            for (int j = 0; j < data.numInstances(); j++) {
                long predTime = System.nanoTime();
                for (int k = 0; k < trainDistributions[j].length; k++)
                    if (oobCounts[j] > 0)
                        trainDistributions[j][k] /= oobCounts[j];
                preds[j] = findIndexOfMax(trainDistributions[j], rand);
                actuals[j] = data.instance(j).classValue();
                predTimes[j] = System.nanoTime() - predTime;
            }
            trainResults.addAllPredictions(actuals, preds, trainDistributions, predTimes, null);
            trainResults.setClassifierName("TSFBagging");
            trainResults.setDatasetName(data.relationName());
            trainResults.setSplit("train");
            trainResults.setFoldID(seed);
            trainResults.finaliseResults(actuals);
            trainResults.setErrorEstimateMethod("OOB");

        }
        //Either do a CV, or bag and get the estimates
        else if (estimator == EstimatorMethod.CV || estimator == EstimatorMethod.NONE) {
            // Defaults to 10 or numInstances, whichever is smaller.
            int numFolds = setNumberOfFolds(data);
            CrossValidationEvaluator cv = new CrossValidationEvaluator();
            if (seedClassifier)
                cv.setSeed(seed * 5);
            cv.setNumFolds(numFolds);
            EnhancedRotationForest rotf = new EnhancedRotationForest();
            rotf.copyParameters(this);
            rotf.setDebug(this.debug);
            if (seedClassifier)
                rotf.setSeed(seed * 100);
            rotf.setEstimateOwnPerformance(false);
//            if (trainTimeContract)//Need to split the contract time, will give time/(numFolds+2) to each fio
//                rotf.setTrainTimeLimit(buildtrainContractTimeNanos / numFolds);
            printLineDebug(" Doing CV evaluation estimate performance with  " + rotf.getTrainContractTimeNanos() / 1000000000 + " secs per fold.");
            long buildTime = trainResults.getBuildTime();
            trainResults = cv.evaluate(rotf, data);
            trainResults.setBuildTime(buildTime);
            trainResults.setClassifierName("RotFCV");
            trainResults.setErrorEstimateMethod("CV_" + numFolds);
        }
        else if (estimator == EstimatorMethod.OOB) {
            // Build a single new TSF using Bagging, and extract the estimate from this
            EnhancedRotationForest rotf = new EnhancedRotationForest();
            rotf.copyParameters(this);
            rotf.setDebug(this.debug);
            rotf.setSeed(seed*33);
            rotf.setEstimateOwnPerformance(true);
            rotf.bagging = true;
//            tsf.setTrainTimeLimit(finalBuildtrainContractTimeNanos);
            printLineDebug(" Doing Bagging estimate performance with " + rotf.getTrainContractTimeNanos() / 1000000000 + " secs per fold ");
            rotf.buildClassifier(data);
            long buildTime = trainResults.getBuildTime();
            trainResults = rotf.trainResults;
            trainResults.setBuildTime(buildTime);
            trainResults.setClassifierName("RotFOOB");
            trainResults.setErrorEstimateMethod("OOB");
        }
    }

    private void copyParameters(EnhancedRotationForest other) {
        this.minNumTrees = other.minNumTrees;
        this.baseClassifier = other.baseClassifier;
        this.minGroup = other.minGroup;
        this.maxGroup = other.maxGroup;
        this.removedPercentage = other.removedPercentage;
        this.maxGroup = other.maxGroup;
        this.minGroup = other.minGroup;
        this.maxGroup = other.maxGroup;


    }


    /**
     * Adds random instances to the dataset.
     *
     * @param dataset the dataset
     * @param numInstances the number of instances
     * @param random a random number generator
     */
    protected void addRandomInstances( Instances dataset, int numInstances,
                                       Random random ) {
        int n = dataset.numAttributes();
        double [] v = new double[ n ];
        for( int i = 0; i < numInstances; i++ ) {
            for( int j = 0; j < n; j++ ) {
                Attribute att = dataset.attribute( j );
                if( att.isNumeric() ) {
                    v[ j ] = random.nextDouble();
                }
                else if ( att.isNominal() ) {
                    v[ j ] = random.nextInt( att.numValues() );
                }
            }
            dataset.add( new DenseInstance( 1, v ) );
        }
    }

    /**
     * Checks minGroup and maxGroup
     *
     * @param data the dataset
     */
    protected void checkMinMax(Instances data) {
        if( minGroup > maxGroup ) {
            int tmp = maxGroup;
            maxGroup = minGroup;
            minGroup = tmp;
        }

        int n = data.numAttributes();
        if( maxGroup >= n )
            maxGroup = n - 1;
        if( minGroup >= n )
            minGroup = n - 1;
    }

    /**
     * Selects a non-empty subset of the classes
     *
     * @param numClasses         the number of classes
     * @param random 	       the random number generator.
     * @return a random subset of classes
     */
    protected boolean [] selectClasses( int numClasses, Random random ) {

        int numSelected = 0;
        boolean selected[] = new boolean[ numClasses ];

        for( int i = 0; i < selected.length; i++ ) {
            if(random.nextBoolean()) {
                selected[i] = true;
                numSelected++;
            }
        }
        if( numSelected == 0 ) {
            selected[random.nextInt( selected.length )] = true;
        }
        return selected;
    }


    /**
     * generates the groups of attributes, given their minimum and maximum
     * sizes.
     *
     * @param data 	the training data to be used for generating the
     * 			groups.
     * @param random 	the random number generator.
     */
    protected int[][] generateGroupFromSize(Instances data, Random random, int maxAtts) {
        int[][] groups;
        int [] permutation = attributesPermutation(data.numAttributes(),
                data.classIndex(), random, maxAtts);

        // The number of groups that have a given size
        int [] numGroupsOfSize = new int[maxGroup - minGroup + 1];

        int numAttributes = 0;
        int numGroups;

        // Select the size of each group
        for( numGroups = 0; numAttributes < permutation.length; numGroups++ ) {
            int n = random.nextInt( numGroupsOfSize.length );
            numGroupsOfSize[n]++;
            numAttributes += minGroup + n;
        }

        groups = new int[numGroups][];
        int currentAttribute = 0;
        int currentSize = 0;
        for( int j = 0; j < numGroups; j++ ) {
            while( numGroupsOfSize[ currentSize ] == 0 )
                currentSize++;
            numGroupsOfSize[ currentSize ]--;
            int n = minGroup + currentSize;
            groups[j] = new int[n];
            for( int k = 0; k < n; k++ ) {
                if( currentAttribute < permutation.length )
                    groups[j][k] = permutation[ currentAttribute ];
                else
                    // For the last group, it can be necessary to reuse some attributes
                    groups[j][k] = permutation[ random.nextInt(
                            permutation.length ) ];
                currentAttribute++;
            }
        }
        return groups;
    }



    final protected int [] attributesPermutation(int numAttributes, int classAttribute,
                                                 Random random, int maxNumAttributes) {
        int [] permutation = new int[numAttributes-1];
        int i = 0;
        //This just ignores the class attribute
        for(; i < classAttribute; i++){
            permutation[i] = i;
        }
        for(; i < permutation.length; i++){
            permutation[i] = i + 1;
        }

        permute( permutation, random );
        if(numAttributes>maxNumAttributes){
            //TRUNCTATE THE PERMATION TO CONSIDER maxNumAttributes.
            // we could do this more efficiently, but this is the simplest way.
            int[] temp = new int[maxNumAttributes];
            System.arraycopy(permutation, 0, temp, 0, maxNumAttributes);
            permutation=temp;
        }
        return permutation;
    }

    /**
     * permutes the elements of a given array.
     *
     * @param v       the array to permute
     * @param random  the random number generator.
     */
    protected void permute( int v[], Random random ) {

        for(int i = v.length - 1; i > 0; i-- ) {
            int j = random.nextInt( i + 1 );
            if( i != j ) {
                int tmp = v[i];
                v[i] = v[j];
                v[j] = tmp;
            }
        }
    }

    /**
     * prints the groups.
     */
    protected void printGroups( ) {
        for( int i = 0; i < groups.size(); i++ ) {
            for( int j = 0; j < groups.get(i).length; j++ ) {
                System.err.print( "( " );
                for( int k = 0; k < groups.get(i)[j].length; k++ ) {
                    System.err.print(groups.get(i)[j][k] );
                    System.err.print( " " );
                }
                System.err.print( ") " );
            }
            System.err.println( );
        }
    }

    /**
     * Transforms an instance for the i-th classifier.
     *
     * @param instance the instance to be transformed
     * @param i the base classifier number
     * @return the transformed instance
     * @throws Exception if the instance can't be converted successfully
     */
    protected Instance convertInstance( Instance instance, int i )
            throws Exception {
        Instance newInstance = new DenseInstance( headers.get(i).numAttributes( ) );
        newInstance.setWeight(instance.weight());
        newInstance.setDataset(headers.get(i));
        int currentAttribute = 0;

        // Project the data for each group
        int[][] g=groups.get(i);
        for( int j = 0; j < g.length; j++ ) {
            Instance auxInstance = new DenseInstance(g[j].length + 1 );
            int k;
            for( k = 0; k < g[j].length; k++ ) {
                auxInstance.setValue( k, instance.value( g[j][k] ) );
            }
            auxInstance.setValue( k, instance.classValue( ) );
            auxInstance.setDataset(reducedHeaders.get(i)[ j ] );
            Filter[] projection=projectionFilters.get(i);
            projection[j].input( auxInstance );
            auxInstance = projection[j].output( );
            projection[j].batchFinished();
            for( int a = 0; a < auxInstance.numAttributes() - 1; a++ ) {
                newInstance.setValue( currentAttribute++, auxInstance.value( a ) );
            }
        }

        newInstance.setClassValue( instance.classValue() );
        return newInstance;
    }

    /**
     * Calculates the class membership probabilities for the given test
     * instance.
     *
     * @param instance the instance to be classified
     * @return predicted class probability distribution
     * @throws Exception if distribution can't be computed successfully
     */
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {

        removeUseless.input(instance);
        instance =removeUseless.output();
        removeUseless.batchFinished();

        normalize.input(instance);
        instance =normalize.output();
        normalize.batchFinished();

        double [] sums = new double [instance.numClasses()], newProbs;

        for (int i = 0; i < classifiers.size(); i++) {
            Instance convertedInstance = convertInstance(instance, i);
            if (instance.classAttribute().isNumeric() == true) {
                sums[0] += classifiers.get(i).classifyInstance(convertedInstance);
            } else {
                newProbs = classifiers.get(i).distributionForInstance(convertedInstance);
                for (int j = 0; j < newProbs.length; j++)
                    sums[j] += newProbs[j];
            }
        }
        if (instance.classAttribute().isNumeric() == true) {
            sums[0] /= (double)classifiers.size();
            return sums;
        } else if (Utils.eq(Utils.sum(sums), 0)) {
            return sums;
        } else {
            Utils.normalize(sums);
            return sums;
        }
    }

    @Override
    public String getParameters() {
        String result="BuildTime,"+trainResults.getBuildTime()+",RemovePercent,"+this.getRemovedPercentage()+",NumFeatures,"+this.getMaxGroup();
        result+=",numTrees,"+numTrees;
        return result;
    }


    @Override //Checkpointable
    public boolean setCheckpointPath(String path) {
        boolean validPath=Checkpointable.super.createDirectories(path);
        if(validPath){
            checkpointPath = path;
            checkpoint = true;
        }
        return validPath;
    }

    @Override
    public void copyFromSerObject(Object obj) throws Exception {
        if(!(obj instanceof EnhancedRotationForest))
            throw new Exception("The SER file is not an instance of ContractRotationForest"); //To change body of generated methods, choose Tools | Templates.
        EnhancedRotationForest saved= ((EnhancedRotationForest)obj);

//Copy RotationForest attributes
        baseClassifier=saved.baseClassifier;
        classifiers=saved.classifiers;
        minGroup = saved.minGroup;
        maxGroup = saved.maxGroup;
        removedPercentage = saved.removedPercentage;
        groups = saved.groups;
        projectionFilter = saved.projectionFilter;
        projectionFilters = saved.projectionFilters;
        headers = saved.headers;
        reducedHeaders = saved.reducedHeaders;
        removeUseless = saved.removeUseless;
        normalize = saved.normalize;


//Copy ContractRotationForest attributes. Not su
        trainResults=saved.trainResults;
        minNumTrees=saved.minNumTrees;
        maxNumTrees=saved.maxNumTrees;
        maxNumAttributes=saved.maxNumAttributes;
        checkpointPath=saved.checkpointPath;
        debug=saved.debug;
        timeUsed=saved.timeUsed;
        numTrees=saved.numTrees;

    }

    /**
     * abstract methods from TrainTimeContractable interface
     * @param amount
     */
    @Override
    public void setTrainTimeLimit(long amount) {
        printLineDebug(" Setting EnhancedRotationForest contract to be "+amount);

        if(amount>0) {
            trainContractTimeNanos = amount;
            trainTimeContract = true;
        }
        else
            trainTimeContract = false;
    }
    @Override
    public boolean withinTrainContract(long start) {
        return start<trainContractTimeNanos;
    }
    @Override
    public long getTrainContractTimeNanos() { return trainContractTimeNanos; }
}

