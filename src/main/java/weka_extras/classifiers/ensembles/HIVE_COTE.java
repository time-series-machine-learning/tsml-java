/*
 * Copyright (C) 2019 xmw13bzu
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package weka_extras.classifiers.ensembles;

import evaluation.evaluators.CrossValidationEvaluator;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import timeseriesweka.classifiers.EnhancedAbstractClassifier;
import timeseriesweka.classifiers.TrainTimeContractable;
import timeseriesweka.classifiers.dictionary_based.BOSS;
import timeseriesweka.classifiers.distance_based.ElasticEnsemble;
import timeseriesweka.classifiers.frequency_based.RISE;
import timeseriesweka.classifiers.hybrids.HiveCote.DefaultShapeletTransformPlaceholder;
import timeseriesweka.classifiers.interval_based.TSF;
import timeseriesweka.classifiers.shapelet_based.ShapeletTransformClassifier;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.Randomizable;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka_extras.classifiers.ensembles.voting.MajorityConfidence;
import weka_extras.classifiers.ensembles.weightings.TrainAcc;

/**
 * TODO jay/tony update javadoc/author list as wanted
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class HIVE_COTE extends AbstractEnsemble implements TechnicalInformationHandler, TrainTimeContractable {

    //TrainTimeContractable
    protected boolean contractingTrainTime = false;
    protected long contractTrainTime = TimeUnit.DAYS.toNanos(7); // if contracting with no time limit given, default to 7 days.
    protected TimeUnit contractTrainTimeUnit = TimeUnit.NANOSECONDS;
    
    
    /**
     * Utility if we want to be conservative while contracting with the overhead 
     * of the ensemble and any variance with the base classifiers' abilities to adhere 
     * to the contract. Give the base classifiers a (very large not not full) proportion
     * of the contract time given, and allow some extra time for the ensemble overhead,
     * potential threading overhead, etc
     */
    protected final double BASE_CLASSIFIER_CONTRACT_PROP = 0.99; //if e.g 1 day contract, 864 seconds grace time
    
    
    
    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation 	result;
        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "J. Lines, S. Taylor and A. Bagnall");
        result.setValue(TechnicalInformation.Field.TITLE, "Time Series Classification with HIVE-COTE: The Hierarchical Vote Collective of Transformation-Based Ensembles");
        result.setValue(TechnicalInformation.Field.JOURNAL, "ACM Transactions on Knowledge Discovery from Data");
        result.setValue(TechnicalInformation.Field.VOLUME, "12");
        result.setValue(TechnicalInformation.Field.NUMBER, "5");
        
        result.setValue(TechnicalInformation.Field.PAGES, "52");
        result.setValue(TechnicalInformation.Field.YEAR, "2018");
        return result;
    }    

    public HIVE_COTE() { 
        super();
    }
    
    @Override
    public void setupDefaultEnsembleSettings() {
        //copied over/adapted from HiveCote.setDefaultEnsembles()
        //TODO jay/tony review
        this.ensembleName = "HIVE-COTE";
        
        this.weightingScheme = new TrainAcc(4);
        this.votingScheme = new MajorityConfidence();
        this.transform = null;
        
        CrossValidationEvaluator cv = new CrossValidationEvaluator(seed, false, false, false, false); 
        cv.setNumFolds(10);
        this.trainEstimator = cv; 

        Classifier[] classifiers = new Classifier[5];
        String[] classifierNames = new String[5];
        
        EnhancedAbstractClassifier ee = new ElasticEnsemble();
        ee.setEstimateOwnPerformance(true);
        classifiers[0] = ee;
        classifierNames[0] = "EE";
        
//        CAWPE st_classifier = new CAWPE();
//        DefaultShapeletTransformPlaceholder st_transform= new DefaultShapeletTransformPlaceholder();
//        st_classifier.setTransform(st_transform);
        ShapeletTransformClassifier st = new ShapeletTransformClassifier();
        if (contractingTrainTime)
            st.setTrainTimeLimit(contractTrainTimeUnit, contractTrainTime);
        classifiers[1] = st;
        classifierNames[1] = "ST";
        
        classifiers[2] = new RISE();
        classifierNames[2] = "RISE";
        
        BOSS boss = new BOSS();
        boss.setEstimateOwnPerformance(true);
        classifiers[3] = boss;
        classifierNames[3] = "BOSS";
        
        TSF tsf = new TSF();
        tsf.setEstimateOwnPerformance(true);
        classifiers[4] = tsf;
        classifierNames[4] = "TSF";
        
        try {
            setClassifiers(classifiers, classifierNames, null);
        } catch (Exception e) {
            System.out.println("Exception thrown when setting up DEFUALT settings of " + this.getClass().getSimpleName() + ". Should "
                    + "be fixed before continuing");
            System.exit(1);
        }
        
        setSeed(seed);
        
        //defaults to 7 day contract TODO jay/tony review
        setTrainTimeLimit(contractTrainTimeUnit, contractTrainTime);
    }

    
    @Override
    public void buildClassifier(Instances data) throws Exception {        
        if (contractingTrainTime) 
            setupContracting();
        
        super.buildClassifier(data);
    }
    
    /**
     * Will split time given evenly among the contractable base classifiers. 
     * 
     * This is currently very naive, and likely innaccurate. Consider these TODOs
     * 
     *  1) If there are any non-contractable base classifiers, these are ignored in 
     *      the contract setting. The full time is allocated among the contractable 
     *      base classifiers, instead of trying to do any wonky guessing of how long the 
     *      non-contractable ones might take
     *  2) Currently, generating accuracy estimates is not considered in the contract.
     *      If there are any non-TrainAccuracyEstimating classifiers, the estimation procedure (e.g.
     *      a 10fold cv) will very likely overshoot the contract, since the classifier would be
     *      trying to keep to contract on each fold and the full build individually, not in total. 
     *      This is an active research question moreso than an implementation question
     *  3) The contract currently does not consider whether the ensemble is being threaded,
     *      i.e. even if it can run the building of two or more classifiers in parallel, 
     *      this will still naively set the contract per classifier as amount/numClassifiers
     */
    @Override //TrainTimeContractable
    public void setTrainTimeLimit(TimeUnit time, long amount) {
        contractingTrainTime = true;
        contractTrainTime = amount;
        contractTrainTimeUnit = time;
    }
    
    /**
     * Sets up the ensemble for contracting, to be called at the start of build classifier,
     * i.e. when parameters can no longer be changed.
     */
    protected void setupContracting() {
        //splits the ensemble contract time between this many classifiers
        int numContractableClassifiers = 0; 
        
        //in future, the number of classifiers we need to separately eval and custom-contract for
        int numNonTrainEstimatingClassifiers = 0; 
        
        for (EnsembleModule module : modules) {
            if(module.isTrainTimeContractable())
                numContractableClassifiers++;
            else 
                System.out.println("WARNING: trying to contract " + ensembleName + ", but base classifier " + module.getModuleName() + " is not contractable, "
                        + "and is therefore not considered in the contract. The ensemble as a whole will very likely not meet the contract.");
            
            if(!module.isAbleToEstimateOwnPerformance()) {
                numNonTrainEstimatingClassifiers++;
                System.out.println("WARNING: trying to contract " + ensembleName + ", but base classifier " + module.getModuleName() + " does not estimate its own accuracy. "
                        + "Performing a separate evaluation on the train set currently is not considered in the contract, and therefore the ensemble as a whole will very "
                        + "likely not meet the contract.");
            }
        }
        
        //force nanos in setting base classifier contracts in case e.g. 1 hour was passed, 1/5 = 0...
        TimeUnit highFidelityUnit = TimeUnit.NANOSECONDS;
        long conservativeBaseClassifierContract = (long) (BASE_CLASSIFIER_CONTRACT_PROP * highFidelityUnit.convert(contractTrainTime, contractTrainTimeUnit));
        long highFidelityTimePerClassifier = (conservativeBaseClassifierContract) / numContractableClassifiers;
        
        for (EnsembleModule module : modules)
            if(module.isTrainTimeContractable())
                ((TrainTimeContractable) module.getClassifier()).setTrainTimeLimit(highFidelityUnit, highFidelityTimePerClassifier);
    }
    
    @Override
    public void setSeed(int seed) { 
        super.setSeed(seed);
        for (EnsembleModule module : modules)
            if(module.getClassifier() instanceof Randomizable)
                ((Randomizable)module.getClassifier()).setSeed(seed);
    }    
    
    public static void main(String[] args) throws Exception {
        System.out.println(ClassifierTools.testUtils_getIPDAcc(new HIVE_COTE()));
    }
}
