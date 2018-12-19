
package vector_classifiers;

import development.MultipleClassifierEvaluation;
import java.util.Arrays;
import java.util.List;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Creates [numClasses] 1vsAll linear regression models. Prediction is max of outputs
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class MultiLinearRegression extends AbstractClassifier {

    Instances numericClassInsts = null;
    LinearRegression[] regressors = null;
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        //creating the 2class version of the insts
        numericClassInsts = new Instances(data);
        numericClassInsts.setClassIndex(0); //temporary
        numericClassInsts.deleteAttributeAt(numericClassInsts.numAttributes()-1);
        Attribute newClassAtt = new Attribute("newClassVal"); //numeric class
        numericClassInsts.insertAttributeAt(newClassAtt, numericClassInsts.numAttributes());
        numericClassInsts.setClassIndex(numericClassInsts.numAttributes()-1); //temporary

        //and building the regressors
        regressors = new LinearRegression[data.numClasses()];
        double[] trueClassVals = data.attributeToDoubleArray(data.classIndex());
        for (int c = 0; c < data.numClasses(); c++) {

            for (int i = 0; i < numericClassInsts.numInstances(); i++) {
                //if this inst is of the class we're currently handling (c), set new class val to 1 else 0
                double cval = trueClassVals[i] == c ? 1 : 0; 
                numericClassInsts.instance(i).setClassValue(cval);
            }    

            regressors[c] = new LinearRegression();
            regressors[c].buildClassifier(numericClassInsts);
        }
    }
    
    @Override
    public double[] distributionForInstance(Instance inst) throws Exception {
        Instances newinst = new Instances(numericClassInsts, 0);
        newinst.add(new DenseInstance(1.0, inst.toDoubleArray()));
        newinst.instance(0).setClassMissing();
        
        double[] outputs = new double[regressors.length];
        for (int i = 0; i < outputs.length; i++) {
            outputs[i] = regressors[i].classifyInstance(newinst.instance(0));
            //class 1 is the class being discriminated FOR
        }
        
        double max = utilities.GenericTools.max(outputs);
        double min = utilities.GenericTools.min(outputs);
        double sum = .0;
        
        //get in range 0 to 1 (may be some negatives)
        if (max == min)
            for (int i = 0; i < outputs.length; i++)
                outputs[i] = 1.0 / outputs.length;//numclasses
        else {
            //get in range 0 to 1, since there may be some negative vals
            //regressing between 0 and 1, something may predict e.g -0.2 
            for (int i = 0; i < outputs.length; i++) {
                outputs[i] = (outputs[i] - min) / (max - min);
                sum += outputs[i];
            }
            
            //and then make them sum to 1
            for (int i = 0; i < outputs.length; i++) 
                outputs[i] /= sum;
        }
        
        return outputs;
    }
    
    @Override
    public double classifyInstance(Instance inst) throws Exception {
        double[] dist = distributionForInstance(inst);
        return utilities.GenericTools.indexOfMax(dist);
    }

    
    public static void main(String[] args) throws Exception {
//                String dsetGroup = "UCI";
//        
//        String basePath = "C:/JamesLPHD/HESCA/"+dsetGroup+"/";
////        String[] datasets = (new File(basePath + "Results/DTWCV/Predictions/")).list();
//        
//        new MultipleClassifierEvaluation(basePath+"XGBoostAnalysis/", dsetGroup+"_testy", 10).
//            setTestResultsOnly(true).
////            setBuildMatlabDiagrams(true).
//            setDatasets(dsetGroup.equals("UCI") ? development.DataSets.UCIContinuousFileNames : development.DataSets.fileNames).
////            setDatasets(basePath + dsetGroup + "2.txt").
//            readInClassifiers(new String[] { "MLR", "1NN", "C4.5", }, basePath+dsetGroup+"Results/").
////            readInClassifiers(new String[] { "XGBoost", "XGBoost500Iterations", "RotF", "RandF" }, basePath+dsetGroup+"Results/").
//            runComparison(); 
        




//        Instances train = ClassifierTools.loadData("Z:/Data/TSCProblems/ItalyPowerDemand/ItalyPowerDemand_TRAIN.arff");
//        Instances test = ClassifierTools.loadData("Z:/Data/TSCProblems/ItalyPowerDemand/ItalyPowerDemand_TEST.arff");
        Instances all = ClassifierTools.loadData("Z:/Data/UCIContinuous/molec-biol-promoter/molec-biol-promoter.arff");
        
        int folds = 10;
        double acc = 0;
        for (int i = 0; i < folds; i++) {
//            Instances[] data = InstanceTools.resampleTrainAndTestInstances(train, test, i);
            Instances[] data = InstanceTools.resampleInstances(all, i, 0.5);
            
            MultiLinearRegression mlr = new MultiLinearRegression();
            mlr.buildClassifier(data[0]);
            double a = .0;
            for (int j = 0; j < data[1].numInstances(); j++) {
                double pred = mlr.classifyInstance(data[1].instance(j));
                double p = data[1].instance(i).classValue();
                if (pred == p)
                    a++;
            }
//            System.out.println(a);
            System.out.println((a/data[1].numInstances()));
            acc+=(a/data[1].numInstances());
        }
        
        System.out.println("acc="+(acc/folds));
    }
}
