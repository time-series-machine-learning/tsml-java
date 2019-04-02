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
package vector_classifiers;

import evaluation.storage.ClassifierResults;
import evaluation.tuning.ParameterResults;
import evaluation.tuning.ParameterSet;
import evaluation.tuning.ParameterSpace;
import evaluation.tuning.Tuner;
import fileIO.OutFile;
import java.util.function.Function;
import timeseriesweka.classifiers.CheckpointClassifier;
import timeseriesweka.classifiers.ContractClassifier;
import timeseriesweka.classifiers.ParameterSplittable;
import timeseriesweka.classifiers.SaveParameterInfo;
import utilities.TrainAccuracyEstimate;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.core.Instance;
import weka.core.Instances;

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
public class TunedClassifier extends AbstractClassifier 
        implements SaveParameterInfo,TrainAccuracyEstimate,SaveEachParameter,ParameterSplittable,CheckpointClassifier,ContractClassifier {

    int seed;
    ParameterSpace space = null;
    Tuner tuner = null;
    AbstractClassifier classifier = null;

    ParameterSet bestParas = null;

    ////////// start interface variables
    
    //we're implementing CheckpointClassifier AND SaveEachParameter for now, however for this classifier checkpointing is 
    //identical to SaveEachParamter, we just implicitely checkpoint after each parameterSet evaluation
    
    String SEP_CP_PS_paraWritePath; //SaveEachParameter //CheckpointClassifier //ParameterSplittable
    boolean SEP_CP_savingAllParameters = false; //SaveEachParameter //CheckpointClassifier
    
    long CC_contractTimeNanos; //ContractClassifier  //note, leaving in nanos for max fidelity, max val of long = 2^64-1 = 586 years in nanoseconds
    boolean CC_contracting = false; //ContractClassifier 
    
    boolean PS_parameterSplitting = false; //ParameterSplittable
    int PS_paraSetID = -1; //ParameterSplittable
    
    //these would refer to the results of the best parameter set
    boolean TAE_writeTrainAcc = false; //TrainAccuracyEstimate
    String TAE_trainAccWritePath; //TrainAccuracyEstimate
    ClassifierResults TAE_trainResults = null; //TrainAccuracyEstimate
    double TAE_trainAcc = -1; //TrainAccuracyEstimate
    
    ////////// end interface variables
    
    
    public TunedClassifier() { 
        this.tuner = new Tuner(); 
    }
    
    public TunedClassifier(AbstractClassifier classifier, ParameterSpace space) { 
        this.classifier = classifier;
        this.space = space;
        this.tuner = new Tuner(); 
    }
    
    public TunedClassifier(AbstractClassifier classifier, ParameterSpace space, Tuner tuner) { 
        this.classifier = classifier;
        this.space = space;
        this.tuner = tuner;
    }
    
    void setSeed(int seed) { 
        this.seed = seed;
        
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
        //setup classifier. in this example, if we wanted to tune the kernal as well, 
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
        if (space == null) { 
            //todo if we end up going with some kind of default para space interface, collect paras from that if classifier is instanceof 
            msg += "Parameter space not setup. ";
            somethingMissing = true;
        }
        if (classifier == null) {
            msg += "No classifier specified. ";
            somethingMissing = true;
        }
        if (somethingMissing) 
            throw new Exception("TunedClassifier: " + msg);
        
        applyInterfaceFlagsToTuner(); //apply any interface flags onto the tuner itself
        
        //special case: if we've been set up to evaluate a particular parameter set in this execution
        //instead of search the full space, evaluate that parameter, write it, and quit
        if (PS_parameterSplitting && PS_paraSetID >= 0) {
            TAE_trainResults = tuner.evaluateParameterSetByIndex(classifier, data, space, PS_paraSetID);
            tuner.saveParaResults(PS_paraSetID, TAE_trainResults);
            return;
            //todo think that's it?
        }
        
        //actual work if normal run
        ParameterResults best = tuner.tune(classifier, data, space);
        
        bestParas = best.paras;
        TAE_trainResults = best.results;
        TAE_trainAcc = best.results.getAcc();
        
        if (TAE_writeTrainAcc && TAE_trainAccWritePath != null)
            TAE_trainResults.writeFullResultsToFile(TAE_trainAccWritePath);
        
        //apply best paras and build final classifier on full train data
        String[] options = best.paras.toOptionsList();
        classifier.setOptions(options);
        classifier.buildClassifier(data);
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

    
    
    
    
    
    
    
    
    
   
    
    
    
    
    
    
    // METHODS FOR:    SaveParameterInfo,TrainAccuracyEstimate,SaveEachParameter,ParameterSplittable,CheckpointClassifier,ContractClassifier
    
    @Override //SaveParameterInfo
    public String getParameters() {
        return getParas(); 
    }

    @Override //TrainAccuracyEstimate
    public void setFindTrainAccuracyEstimate(boolean setCV) {
        System.out.println("-------Inside setFindTrainAccuracyEstimate(..), but for TunedClassifier this is redundant");
        //do nothing
    }

    @Override //TrainAccuracyEstimate
    public void writeCVTrainToFile(String train) {
        this.TAE_trainAccWritePath = train;
        this.TAE_writeTrainAcc = true;
    }

    @Override //TrainAccuracyEstimate
    public ClassifierResults getTrainResults() {
        return TAE_trainResults;
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

    @Override //ParameterSplittable
    public String getParas() {
        return bestParas.toClassifierResultsParaLine(true);
    }

    @Override //ParameterSplittable
    public double getAcc() {
        throw new UnsupportedOperationException("-------Dont think this is needed anywhere; testing. Part of the ParameterSplittable interface"); 
    }

    @Override //CheckpointClassifier
    public void setSavePath(String path) {
        this.SEP_CP_PS_paraWritePath = path;
        this.SEP_CP_savingAllParameters = true;
    }

    @Override //CheckpointClassifier
    public void copyFromSerObject(Object obj) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override //ContractClassifier
    public void setTimeLimit(long time) {
        CC_contracting = true;
        CC_contractTimeNanos = time;
    }

    @Override //ContractClassifier
    public void setTimeLimit(TimeLimit time, int amount) {
        CC_contracting = true;
        long secToNano = 1000000000L;
        
        switch(time){
            case MINUTE:
                CC_contractTimeNanos = amount*60*secToNano;
                break;
            case HOUR: default:
                CC_contractTimeNanos= amount*60*60*secToNano;
                break;
            case DAY:
                CC_contractTimeNanos= amount*24*60*60*secToNano; 
                break;
        }
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
        
        if (CC_contracting)
            tuner.setTimeLimit(this.CC_contractTimeNanos);
    }
}
