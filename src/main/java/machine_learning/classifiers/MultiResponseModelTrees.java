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
package machine_learning.classifiers;

import experiments.data.DatasetLoading;
import utilities.InstanceTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.trees.M5P;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Creates [numClasses] 1vsAll model trees (M5). Prediction is max of outputs
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class MultiResponseModelTrees extends AbstractClassifier {

    Instances numericClassInsts = null;
    M5P[] regressors = null;
    
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
        regressors = new M5P[data.numClasses()];
        double[] trueClassVals = data.attributeToDoubleArray(data.classIndex());
        for (int c = 0; c < data.numClasses(); c++) {

            for (int i = 0; i < numericClassInsts.numInstances(); i++) {
                //if this inst is of the class we're currently handling (c), set new class val to 1 else 0
                double cval = trueClassVals[i] == c ? 1 : 0; 
                numericClassInsts.instance(i).setClassValue(cval);
            }    

            regressors[c] = new M5P();
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
//        Instances train = ClassifierTools.loadDataThrowable("Z:/Data/TSCProblems/ItalyPowerDemand/ItalyPowerDemand_TRAIN.arff");
//        Instances test = ClassifierTools.loadDataThrowable("Z:/Data/TSCProblems/ItalyPowerDemand/ItalyPowerDemand_TEST.arff");
        Instances all = DatasetLoading.loadDataNullable("Z:/Data/UCIContinuous/hayes-roth/hayes-roth.arff");
        
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
