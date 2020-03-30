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
package machine_learning.classifiers.tuned;

import evaluation.evaluators.CrossValidationEvaluator;
import evaluation.evaluators.InternalEstimateEvaluator;
import evaluation.tuning.ParameterResults;
import evaluation.tuning.ParameterSet;
import evaluation.tuning.ParameterSpace;
import evaluation.tuning.Tuner;
import tsml.classifiers.ParameterSplittable;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;

import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.Checkpointable;
import tsml.classifiers.TrainTimeContractable;
import machine_learning.classifiers.SaveEachParameter;
import tsml.classifiers.Tuneable;

/**
 * Given 
 *      1) a tuning method, 
 *              - DEFAULT: tuner set to gridsearch + 10foldcv 
 *              - settable via setTuner(...)
 *      2) a base classifier with a well formed setOptions(String[]) method 
 *          (that must extend AbstractClassifier, the Classifier interface 
 *          alone does not provide setOptions()) 
 *              - DEFAULT: none
 *              - settable via setClassifier(...)
 *      3) a parameter space expressed as keys (that correspond to flags in the setOptions() method) 
 *          to lists of values that these parameters can take.
 *              - DEFAULT: none
 *              - settable via setParameterSpace(...)
 * 
 * For a basic example of the above, see setupTestTunedClassifier()
 * 
 * This class will select the best parameter set on the given dataset according to the 
 * selection and evaluation methods described by the tuner, and build the base classifier
 * with the best parameters found
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class TunedClassifier extends EnhancedAbstractClassifier 
        implements SaveEachParameter,ParameterSplittable,Checkpointable, TrainTimeContractable {

    ParameterSpace space = null;
    Tuner tuner = null;
    AbstractClassifier classifier = null;

    ParameterSet bestParas = null;
    String[] bestOptions = null;

    ////////// start interface variables
    
    //we're implementing CheckpointClassifier AND SaveEachParameter for now, however for this classifier checkpointing is 
    //identical to SaveEachParamter, we just implicitely checkpoint after each parameterSet evaluation
    
    String SEP_CP_PS_paraWritePath; //SaveEachParameter //CheckpointClassifier //ParameterSplittable
    boolean SEP_CP_savingAllParameters = false; //SaveEachParameter //CheckpointClassifier
    
    long trainContractTimeNanos; //TrainTimeContractClassifier  //note, leaving in nanos for max fidelity, max val of long = 2^64-1 = 586 years in nanoseconds
    boolean trainTimeContract = false; //TrainTimeContractClassifier
    
    boolean PS_parameterSplitting = false; //ParameterSplittable
    int PS_paraSetID = -1; //ParameterSplittable
    ////////// end interface variables

    /**
     * Creates an empty TunedClassifier. Tuner has a default value, however at minimum the classifier and parameter space
     * shall need to be provided later via set...() methods
     */
    public TunedClassifier() { 
        this(null, null, new Tuner());
    }

    /**
     * If the classifier is able to estimate its own performance while building, the tuner shall default to using that
     * as the evaluation method. Otherwise defaults to an external 10fold cv
     */
    public TunedClassifier(AbstractClassifier classifier, ParameterSpace space) {
        this(classifier, space,
            EnhancedAbstractClassifier.classifierAbleToEstimateOwnPerformance(classifier) ?
                    new Tuner(new InternalEstimateEvaluator()) :
                    new Tuner(new CrossValidationEvaluator())
            );
    }
    
    public TunedClassifier(AbstractClassifier classifier, ParameterSpace space, Tuner tuner) { 
        super(CAN_ESTIMATE_OWN_PERFORMANCE);
        this.classifier = classifier;
        this.space = space;
        this.tuner = tuner;
    }

    /**
     * PRE: Classifier must be set, if not, noothing happens
     * @return true if successful in turning on internal estimate
     */

    public boolean useInternalEstimates(){
            if(classifier==null)
                return false;
           if(EnhancedAbstractClassifier.classifierAbleToEstimateOwnPerformance(classifier) ){
               tuner=new Tuner(new InternalEstimateEvaluator());
               return true;
           }
           return false;
    }
    public void setSeed(int seed) { 
        super.setSeed(seed);
        tuner.setSeed(seed);
        //no setSeed in abstractclassifier. i imagine most define it via setOptions,
        //so could add it a a parameter with only one possible value, or jsut set the seed
        //before giving the classifier to this tunedclassifier instance
    }
    
    public boolean getCloneClassifierForEachParameterEval() {
        return tuner.getCloneClassifierForEachParameterEval();
    }

    public void setCloneClassifierForEachParameterEval(boolean clone) {
        tuner.setCloneClassifierForEachParameterEval(clone);
    }

    public String[] getBestOptions() { return bestOptions; }

    public ParameterSpace getSpace() {
        return space;
    }

    public void setParameterSpace(ParameterSpace space) {
        this.space = space;
    }

    public Tuner getTuner() {
        return tuner;
    }

    public void setTuner(Tuner tuner) {
        this.tuner = tuner;
    }

    public AbstractClassifier getClassifier() {
        return classifier;
    }

    public void setClassifier(AbstractClassifier classifier) {
        this.classifier = classifier;
    }

    public void setupTestTunedClassifier() {         
        //setup classifier. in this example, if we wanted to tune the kernel as well, 
        //we'd have to extend smo and override the setOptions to allow specific options
        //for kernal settings... see SMO.setOptions()
        SMO svm = new SMO();
        PolyKernel p=new PolyKernel();
        p.setExponent(2);
        svm.setKernel(p);
        this.classifier = new SMO();                
        
        //setup tuner, defaults to 10foldCV grid-search
        this.tuner = new Tuner(); 
        
        //setup para space 
        int size = 13;
        double[] cs = new double[size];
        for (int i = 0; i < cs.length; i++)
            cs[i] = Math.pow(10.0, (i-size/2));
        
        this.space = new ParameterSpace();
        this.space.addParameter("C", cs);
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        
        //check everything's here/init
        boolean somethingMissing = false;
        String msg = "";
        if (tuner == null) {
            msg += "Tuner not setup. ";
            somethingMissing = true;
        }
        if (classifier == null) {
            msg += "No classifier specified. ";
            somethingMissing = true;
        }
        if (space == null) { 
            if (classifier instanceof Tuneable)
                space = ((Tuneable)classifier).getDefaultParameterSearchSpace();
            else {
                msg += "Parameter space not setup. ";
                somethingMissing = true;
            }
        }
        if (somethingMissing) 
            throw new Exception("TunedClassifier: " + msg);
        
        applyInterfaceFlagsToTuner(); //apply any interface flags onto the tuner itself
        
        //special case: if we've been set up to evaluate a particular parameter set in this execution
        //instead of search the full space, evaluate that parameter, write it, and quit
        if (PS_parameterSplitting && PS_paraSetID >= 0) {
            trainResults = tuner.evaluateParameterSetByIndex(classifier, data, space, PS_paraSetID);
            tuner.saveParaResults(PS_paraSetID, trainResults);
            return;
            //todo think that's it?
        }
        
        //actual work if normal run
        ParameterResults best = tuner.tune(classifier, data, space);
        
        bestParas = best.paras;
        trainResults = best.results;
        
        //apply best paras and build final classifier on full train data
        String[] options = best.paras.toOptionsList();
        bestOptions = Arrays.copyOf(options, options.length);
        classifier.setOptions(options);
        classifier.buildClassifier(data);
        trainResults.setParas(getParameters());
    }
 
    @Override
    public double[] distributionForInstance(Instance inst) throws Exception { 
        return classifier.distributionForInstance(inst);
    }
    
    public static void main(String[] args) throws Exception {
        
//        String dataset = "hayes-roth";
//        
//        TunedClassifier tcGrid = new TunedClassifier();
//        tcGrid.setupTestTunedClassifier();
//        tcGrid.setCloneClassifierForEachParameterEval(false);
//        
//        TunedClassifier tcRand = new TunedClassifier();
//        tcRand.setupTestTunedClassifier();
//        tcRand.getTuner().setSearcher(new RandomSearcher(3));
//        tcRand.getTuner().setEvaluator(new StratifiedResamplesEvaluator());
//        tcRand.setCloneClassifierForEachParameterEval(false);
//        
//        
//        Classifier[] cs = new Classifier[] { tcRand, new SMO(), tcGrid };
//        
//        int numFolds = 10;
//        
//        for (Classifier c : cs) {
//            Instances all = ClassifierTools.loadData("Z:\\Data\\UCIDelgado\\"+dataset+"\\"+dataset+".arff");
//            double mean =.0;
//            
//            for (int f = 0; f < numFolds; f++) {
//                Instances[] data = InstanceTools.resampleInstances(all, f, 0.5);
//                
//                try {
//                    ((TunedClassifier)c).setSeed(f);
//                }catch (Exception e){ }
//                     
//                c.buildClassifier(data[0]);
//                double t = ClassifierTools.accuracy(data[1], c);
//                mean += t;
//                System.out.print(t + ", ");
//            }
//            
//            mean /= numFolds;
//            System.out.println("\nmean = " + mean);
//        }
        

        experiments.Experiments.ExperimentalArguments exp = new experiments.Experiments.ExperimentalArguments();
        exp.checkpointing = true;
        exp.classifierName = "TunedSMO";
        exp.datasetName = "hayes-roth";
        exp.foldId = 1;
        exp.generateErrorEstimateOnTrainSet = true;
        exp.dataReadLocation = "Z:\\Data\\UCIDelgado\\";
        exp.resultsWriteLocation = "C:\\Temp\\TunerTests\\t\\";
//        
//        exp.singleParameterID = 1;
        
        experiments.Experiments.setupAndRunExperiment(exp);
    }

    
    
    
    
    
    
    
    
    
   
    
    
    
    
    
    
    // METHODS FOR:    TrainAccuracyEstimate,SaveEachParameter,ParameterSplittable,CheckpointClassifier,TrainTimeContractClassifier
    
    @Override
    public String getParameters() {
        String str=classifier.getClass().getSimpleName();
        if(classifier instanceof EnhancedAbstractClassifier)
            str+=","+((EnhancedAbstractClassifier)classifier).getParameters();
        return str;
    }

    @Override //SaveEachParameter
    public void setPathToSaveParameters(String r) {
        this.SEP_CP_PS_paraWritePath = r;
        this.SEP_CP_savingAllParameters = true;
    }

    @Override //SaveEachParameter
    public void setSaveEachParaAcc(boolean b) {
        this.SEP_CP_savingAllParameters = b;
    }

    @Override //ParameterSplittable
    public void setParamSearch(boolean b) {
        throw new UnsupportedOperationException("-------This was intended to turn off the tuning "
                + "of parameters while evaluating a particular parameter set in the original tuned classifiers. "
                + "Now that we're in a general tunedClassifier specifically, this doesnt make sense. Part of the ParameterSplittable interface"); 
    }

    @Override //ParameterSplittable
    public void setParametersFromIndex(int x) {
        this.PS_paraSetID = x;
        this.PS_parameterSplitting = true;
    }

 //   @Override //ParameterSplittable
 //   public String getParas() {
 //       return bestParas.toClassifierResultsParaLine(true);
 //   }

    @Override //Checkpointable
    public boolean setCheckpointPath(String path) {
        boolean validPath=Checkpointable.super.createDirectories(path);
        if(validPath){
            this.SEP_CP_PS_paraWritePath = path;
            this.SEP_CP_savingAllParameters = true;
        }
        return validPath;
    }

    @Override //CheckpointClassifier
    public void copyFromSerObject(Object obj) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void setTrainTimeLimit(long amount) {
        trainContractTimeNanos =amount;
        trainTimeContract = true;
    }

    
    /**
     * To be called at start of buildClassifier
     * 
     * Simple helper method to transfer necessary interface variable changes over to the tuner
     * in case user e.g sets up interface variables THEN sets a new tuner, or sets a new tuner
     * (or sticks with default) THEN sets these variables, etc
     */
    private void applyInterfaceFlagsToTuner() {
        if (SEP_CP_savingAllParameters || PS_parameterSplitting)
            tuner.setPathToSaveParameters(this.SEP_CP_PS_paraWritePath);
        
        if (trainTimeContract)
            tuner.setTrainTimeLimit(this.trainContractTimeNanos);
    }
}
