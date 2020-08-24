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
package tsml.classifiers;

import weka.classifiers.AbstractClassifier;
import evaluation.storage.ClassifierResults;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import evaluation.storage.ClassifierResults;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.*;

/**
 *
 * Extends the AbstractClassifier to achieve the following:
 * 1. Allow storage of information about the training process, including timing info, 
 * any optimisation performed and any train set estimates and predictions made to help 
 * assess generalisability. See below for more info on this major enhancement.
 * 2. Allow for a unified process of seeding classifiers. The seed is stored in seed, 
 * and set via the interface Randomizable
 * 3. Allow for default getCapapabilities() for TSC. For time series, these default to all real
 * valued attributes, no missing values, and classification only. This overrides default
 * behaviour in AbstractClassifier
 * 4. Allow for standardised mechanism for saving classifier information to file. 
 * For example usage, see the classifier TSF.java.  the method getParameters() 
 * can be enhanced to include any parameter info for the final classifier. 
 * getParameters() is called to store information on the second line of file 
 * storage format testFoldX.csv.
 
Train data is the major enhancement: There are two components: time taken in training, and any 
train set predictions produced internally. 

*there are three components to the time that may be spent building a classifier

* 1. timing
buildTime
* the minimum any classifier that extends this should store
 is the build time in buildClassifier, through calls to System.currentTimeMillis()
 or nanoTime() at the start and end of the method, stored in trainResults, with
 trainResults.setBuildTime(totalBuildTime) nanoTime() is generally preferred, and 
 to set the TimeUnit of the ClassiiferReults object appropriately, e.g 
 trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
errorEstimateTime
* the exact usage of this statistic has not been finalised. Conceptually measures
* how long is spent estimating the test error from the train data
buildPlusEstimateTime
 * 2. Recording train set results
ClassifierResults trainResults can also store other information about the training,
 including estimate of accuracy, predictions and probabilities. The mechanism for finding
 these is algorithm specific. They key point is that  all values in trainResults are 
 set without any reference to the train set at all. All the variables for trainResults 
 are set in buildClassifier, which has no access to test data at all. It is completely decoupled. 
 
 Instances train=//Get train
 
 EnhancedAbstractClassifier c= //Get classifier
 c.buildClassifier(train)    //ALL STATS SET HERE
 * Update 1/7/2020:
 * @author Tony Bagnall and James Large EstimatorMethod estimator moved up from subclasses, since the pattern
 * appears in multiple forest based ensembles
 */
abstract public class EnhancedAbstractClassifier extends AbstractClassifier implements SaveParameterInfo,
                                                                                       Serializable,
                                                                                       Randomizable,
                                                                                       TSClassifier {

/** Store information of training. The minimum should be the build time, tune time and/or estimate acc time      */
    protected ClassifierResults trainResults = new ClassifierResults();
    protected int seed = 0;
    /**Can seed for reproducibility*/
    protected Random rand=new Random(seed);
    protected boolean seedClassifier=false;
    protected transient boolean debug=false;

    /**
     * get the classifier RNG	
     * @return Random
     */
    public Random getRandom() {
        return rand;
    }

    public AbstractClassifier getClassifier(){
        return this;
    }

    /**
     * Set the classifier RNG	
     * @param rand
     */
    public void setRandom(Random rand) {
        this.rand = rand;
    }
    
    /**
     * A printing-friendly and/or context/parameter-aware name that can optionally
     * be used to describe this classifier. By default, this will simply be the
     * simple-class-name of the classifier
     */
    protected String classifierName = getClass().getSimpleName();

    /**
     * This flags whether classifiers are able to estimate their own performance
     * (possibly with some bias) on the train data in some way as part of their buildClassifier
     * fit, and avoid an external fully nested-cross validation process.
     *
     * This flag being true indicates the ABILITY to estimate train performance,
     * to turn this behaviour on, setEstimateOwnPerformance(true) should be called.
     * By default, the estimation behaviour is off regardless of ability
     *
     * This way, unnecessary work is avoided and if for whatever reason a nested
     * estimation process is explicitly wanted (e.g. for completely bias-free estimates),
     * that can also be achieved.
     *
     * This variable is private and only settable via the abstract constructor,
     * such that all subclasses must set it at initialisation.
     *
     * This variable and the related gets/sets replace the TrainAccuracyEstimator interface
     */
    protected boolean ableToEstimateOwnPerformance = false;

    /**
     * This flags whether the classifier shall estimate their own performance
     * (possibly with some bias) on the train data in some way as part of their buildClassifier
     * fit, and avoid a full nested-cross validation process.
     *
     * The estimation process may be entirely encapsulated in the build process (e.g. a tuned
     * classifier returning the train estimate of the best parameter set, acting as the train
     * estimate of the full classifier: note the bias), or may be done as an
     * additional step beyond the normal build process but far more efficiently than a
     * nested cv (e.g. a 1NN classifier could perform an efficient internal loocv)
     */
    protected boolean estimateOwnPerformance = false;

    /** If trainAccuracy is required, there are two options that can be implemented
     *   1. estimator=CV: do a 10x CV on the train set with a clone of this classifier
     *   2. estimator=OOB: build an OOB model just to get the OOB accuracy estimate
     */
    public enum EstimatorMethod{CV,OOB,NONE}
    protected EstimatorMethod estimator=EstimatorMethod.NONE;
    public void setEstimatorMethod(String str){
        String s=str.toUpperCase();
        if(s.equals("CV"))
            estimator=EstimatorMethod.CV;
        else if(s.equals("OOB"))
            estimator=EstimatorMethod.OOB;
        else if(s.equals("NONE")) {
            estimator = EstimatorMethod.NONE;
        }
        else
            throw new UnsupportedOperationException("Unknown estimator method in classifier "+getClass().getSimpleName()+" = "+str);
    }

    public String getEstimatorMethod() {
        return estimator.name();
    }

    //utilities for readability in setting the above bools via super constructor in subclasses
    public static final boolean CAN_ESTIMATE_OWN_PERFORMANCE = true;
    public static final boolean CANNOT_ESTIMATE_OWN_PERFORMANCE = false;
    protected int numClasses = -1;

    public int getNumClasses() {
        return numClasses;
    }

    protected void setAbleToEstimateOwnPerformance(boolean state) {
        ableToEstimateOwnPerformance = state;
    }

    @Override
    public void buildClassifier(final Instances trainData) throws
                                                                Exception {
        trainResults = new ClassifierResults();
        rand.setSeed(seed);
        numClasses = trainData.numClasses();
        trainResults.setClassifierName(getClassifierName());
        trainResults.setParas(getParameters());
        if(trainData.classIndex() != trainData.numAttributes() - 1) {
            throw new IllegalArgumentException("class value not at the end");
        }
    }

    public EnhancedAbstractClassifier() {
        this(false);
    }

    public EnhancedAbstractClassifier(boolean ableToEstimateOwnPerformance) {
        this.ableToEstimateOwnPerformance = ableToEstimateOwnPerformance;
        setDebug(debug);
    }

    @Override
    public int hashCode() {
        if(classifierName == null) {
            return super.hashCode();
        }
        return classifierName.hashCode();
    }

    @Override
    public boolean equals(Object other) {
        if(!(other instanceof EnhancedAbstractClassifier)) {
            return false;
        }
        EnhancedAbstractClassifier eac = (EnhancedAbstractClassifier) other;
        return classifierName.equalsIgnoreCase(eac.classifierName);
    }
    
    /**
     * This flags whether the classifier shall estimate their own performance 
     * (possibly with some bias) on the train data in some way as part of their buildClassifier
     * fit, and avoid a full nested-cross validation process.
     * 
     * The estimation process may be entirely encapsulated in the build process (e.g. a tuned 
     * classifier returning the train estimate of the best parameter set, acting as the train 
     * estimate of the full classifier: note the bias), or may be done as an
     * additional step beyond the normal build process but far more efficiently than a
     * nested cv (e.g. a 1NN classifier could perform an efficient internal loocv, or a tree ensemble 
     * can use out of bag estimates)  
     */
    public boolean ableToEstimateOwnPerformance() { 
        return ableToEstimateOwnPerformance;
    }
    
    /**
     * This flags whether the classifier shall estimate their own performance 
     * (possibly with some bias) on the train data in some way as part of their buildClassifier
     * fit, and avoid a full nested-cross validation process.
     * 
     * The estimation process may be entirely encapsulated in the build process (e.g. a tuned 
     * classifier returning the train estimate of the best parameter set, acting as the train 
     * estimate of the full classifier: note the bias), or may be done as an
     * additional step beyond the normal build process but far more efficiently than a
     * nested cv (e.g. a 1NN classifier could perform an efficient internal loocv)  
     */
    public void setEstimateOwnPerformance(boolean estimateOwnPerformance) throws IllegalArgumentException {
        if (estimateOwnPerformance && !ableToEstimateOwnPerformance)
            throw new IllegalArgumentException("Classifier ("+getClassifierName()+") is unable to estimate own performance, but "
                    + "trying to set it to do so. Check with ableToEstimateOwnPerformance() first");
        this.estimateOwnPerformance = estimateOwnPerformance;
    }
    
    /**
     * This flags whether the classifier shall estimate their own performance 
     * (possibly with some bias) on the train data in some way as part of their buildClassifier
     * fit, and avoid a full nested-cross validation process.
     * 
     * The estimation process may be entirely encapsulated in the build process (e.g. a tuned 
     * classifier returning the train estimate of the best parameter set, acting as the train 
     * estimate of the full classifier: note the bias), or may be done as an
     * additional step beyond the normal build process but far more efficiently than a
     * nested cv (e.g. a 1NN classifier could perform an efficient internal loocv)  
     */
    public boolean getEstimateOwnPerformance() {
        return estimateOwnPerformance;
    }
    
    /**
     * A simple utility to wrap the test of whether a classifier reference contains an
     * EnhancedAbstractClassifier object, and whether that classifier CAN estimate 
     * its own accuracy internally. 
     * 
     * Replacing the previous test 'classifier instanceof TrainAccuracyEstimator'
     */
    public static boolean classifierAbleToEstimateOwnPerformance(Classifier classifier) { 
        return classifier instanceof EnhancedAbstractClassifier && 
                    ((EnhancedAbstractClassifier) classifier).ableToEstimateOwnPerformance(); 
    }
    
    /**
     * A simple utility to wrap the test of whether a classifier reference contains an
     * EnhancedAbstractClassifier object, and whether that classifier has been set up 
     * to estimate its own accuracy internally. 
     * 
     * Replacing the previous test 'classifier instanceof TrainAccuracyEstimator''
     */
    public static boolean classifierIsEstimatingOwnPerformance(Classifier classifier) { 
        return classifier instanceof EnhancedAbstractClassifier && 
                    ((EnhancedAbstractClassifier) classifier).getEstimateOwnPerformance(); 
    }
       
    
    @Override
    public String getParameters() {
        return "seedClassifier,"+seedClassifier+",seed,"+seed;
    }
     
    /**
     * Gets the train results for this classifier, which will be empty (but not-null) 
     * until buildClassifier has been called.
     * 
     * If the classifier has ableToEstimateOwnPerformance()==true and was set-up to estimate 
     * it's own train accuracy (setEstimateOwnPerformance(true) called), these will be populated 
     * with full prediction information, ready to be written as a trainFoldX file for example
     * 
     * Otherwise, the object will at minimum contain the build time, classifiername, 
     * and parameter information
     */
    public ClassifierResults getTrainResults() {
        return trainResults;
    }
    
    /**
     * Set the seed for random number generation. Re-initialises internal RNG 
     * with new seed. Therefore, note that in general this method should be called
     * before the classifier needs to use the RNG, and that in general the RNG 
     * shouldn't ever be used in the constructor
     * 
     * @param seed the seed 
     */
    @Override
    public void setSeed(int seed) { 
        seedClassifier=true;
        this.seed = seed;
        rand=new Random(seed);
    }

    /**
     * Gets the seed for the random number generations
     *
     * @return the seed for the random number generation
     */
    @Override
    public int getSeed() {
        return seed;
    }
    
    /**
     * Returns default capabilities of the classifier. These are that the 
     * data must be numeric, with no missing and a nominal class
     * @return the capabilities of this classifier
     */    
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        // attributes must be numeric
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        // Can only handle discrete class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        // instances
        result.setMinimumNumberInstances(2);
        return result;
    }
    
    public int setNumberOfFolds(Instances data){
        return data.numInstances()<10?data.numInstances():10;
    }    
    
    /**
     * A printing-friendly and/or context/parameter-aware name that can optionally 
     * be used to describe this classifier. By default, this will simply be the 
     * simple-class-name of the classifier.
     */
    public String getClassifierName() {
        if (classifierName == null)
            classifierName = this.getClass().getSimpleName();
        return classifierName;
    }

    /**
     * Method to find the best index in a list of doubles. If a tie occurs,
     * then the best index is chosen randomly.
     *
     * @param x a list of doubles.
     * @param rand a Random object.
     * @return the index of the highest value in x.
     */
    public static int findIndexOfMax(double [] x, Random rand) {
        double currentMax = x[0];
        ArrayList<Integer> bestIndexes = new ArrayList<>();
        bestIndexes.add(0);

        //Find the best index(es)
        for(int i = 1; i < x.length; i++) {
            if(x[i] > currentMax) {
                bestIndexes.clear();
                bestIndexes.add(i);
                currentMax = x[i];
            } else if(x[i] == currentMax) {
                bestIndexes.add(i);
            }
        }

        //No ties occured
        if(bestIndexes.size() == 1) {
            return bestIndexes.get(0);
        } else {
            //ties did occur
            return bestIndexes.get(rand.nextInt(bestIndexes.size()));
        }
    }

    /**
     * Method to find the best index in a list of doubles. If a tie occurs,
     * then the best index is chosen randomly.
     *
     * @param x a list of doubles.
     * @param seed a long seed for a Random object.
     * @return the index of the highest value in x.
     */
    public static int findIndexOfMax(double [] x, long seed) {
        double currentMax = x[0];
        ArrayList<Integer> bestIndexes = new ArrayList<>();
        bestIndexes.add(0);

        //Find the best index(es)
        for(int i = 1; i < x.length; i++) {
            if(x[i] > currentMax) {
                bestIndexes.clear();
                bestIndexes.add(i);
                currentMax = x[i];
            } else if(x[i] == currentMax) {
                bestIndexes.add(i);
            }
        }

        //No ties occured
        if(bestIndexes.size() == 1) {
            return bestIndexes.get(0);
        } else {
            //ties did occur
            return bestIndexes.get(new Random(seed).nextInt(bestIndexes.size()));
        }
    }

    /**
     * Overrides default AbstractClassifier classifyInstance to use random tie breaks.
     * Classifies the given test instance. The instance has to belong to a
     * dataset when it's being classified. Note that a classifier MUST
     * implement either this or distributionForInstance().
     *
     * @param instance the instance to be classified
     * @return the predicted most likely class for the instance or
     * Utils.missingValue() if no prediction is made
     * @exception Exception if an error occurred during the prediction
     */
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double [] dist = distributionForInstance(instance);
        if (dist == null) {
            throw new Exception("Null distribution predicted");
        }
        switch (instance.classAttribute().type()) {
            case Attribute.NOMINAL:
                return findIndexOfMax(dist, rand);
            case Attribute.NUMERIC:
                return dist[0];
            default:
                return Utils.missingValue();
        }
    }
    
    /**
     * Sets a printing-friendly and/or context/parameter-aware name that can optionally 
     * be used to describe this classifier. By default, this will simply be the 
     * simple-class-name of the classifier
     */
    public void setClassifierName(String classifierName) {
        this.classifierName = classifierName;
    }

    public void setDebug(boolean b){
        debug=b;
    }

    public boolean isDebug() {
        return debug;
    }

    public String toString() {
        return getClassifierName();
    }

    public void printDebug(String s){
        if(debug)
            System.out.print(s);
    }
    public void printLineDebug(String s){
        if(debug)
            System.out.println(s);
    }
}
