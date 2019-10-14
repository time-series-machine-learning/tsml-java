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
package timeseriesweka.classifiers;

import weka.classifiers.AbstractClassifier;
import evaluation.storage.ClassifierResults;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instances;
import weka.core.Randomizable;

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
* 
 
 * 2. Recording train set results
ClassifierResults trainResults can also store other information about the training,
 including estimate of accuracy, predictions and probabilities. The mechanism for finding
 these is algorithm specific. They key point is that  all values in trainResults are 
 set without any reference to the train set at all. All the variables for trainResults 
 are set in buildClassifier, which has no access to test data at all. It is completely decoupled. 
 
 Instances train=//Get train
 
 EnhancedAbstractClassifier c= //Get classifier
 c.buildClassifier(train)    //ALL STATS SET HERE
 * 
 * @author Tony Bagnall and James Large
 */
abstract public class EnhancedAbstractClassifier extends AbstractClassifier implements SaveParameterInfo, Randomizable {
        
/** Store information of training. The minimum should be the build time, tune time and/or estimate acc time      */
    protected ClassifierResults trainResults =new ClassifierResults();
/**Can seed for reproducibility*/
    protected Random rand=new Random();
    protected boolean seedClassifier=false;
    protected int seed = 0;
    
    /**
     * A printing-friendly and/or context/parameter-aware name that can optionally 
     * be used to describe this classifier. By default, this will simply be the 
     * simple-class-name of the classifier
     */
    protected String classifierName = null;
    
    /**
     * This flags whether classifiers are able to estimate their own performance 
     * (possibly with some bias) on the train data in some way as part of their buildClassifier
     * fit, and avoid an external fully nested-cross validation process.
     * 
     * This flag being true indicates the ABILITY to estimate train performance, 
     * to turn this behaviour on, setEstimateOwnPerformance(true) should be called. 
     * By default, the estimation behaviour is off regardless of ability
     * 
     * This way, if for whatever reason a nested estimation process is explicitly wanted 
     * (e.g. for completely bias-free estimates), that can also be achieved
     * 
     * This variable is private and only settable via the abstract constructor, 
     * such that all subclasses must set it at initialisation.
     * 
     * This variable and the related gets/sets replace the TrainAccuracyEstimator interface
     */
    boolean ableToEstimateOwnPerformance = false;
    
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
    boolean estimateOwnPerformance = false;
    
    //utilities for readability in setting the above bools via super constructor in subclasses
    public static final boolean CAN_ESTIMATE_OWN_PERFORMANCE = true;
    public static final boolean CANNOT_ESTIMATE_OWN_PERFORMANCE = false;
    
    public EnhancedAbstractClassifier(boolean ableToEstimateOwnPerformance) {
        this.ableToEstimateOwnPerformance = ableToEstimateOwnPerformance;
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
        trainResults.setClassifierName(getClassifierName());
        trainResults.setParas(getParameters());
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
//        rand.setSeed(seed);
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
        // Here add in relational when ready
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        // instances
        result.setMinimumNumberInstances(0);   
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
     * Sets a printing-friendly and/or context/parameter-aware name that can optionally 
     * be used to describe this classifier. By default, this will simply be the 
     * simple-class-name of the classifier
     */
    public void setClassifierName(String classifierName) {
        this.classifierName = classifierName;
    }
    protected boolean debug=false;
    public void setDebug(boolean b){
        debug=b;
    }

}
