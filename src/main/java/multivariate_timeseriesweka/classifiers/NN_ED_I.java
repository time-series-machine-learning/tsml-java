/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package multivariate_timeseriesweka.classifiers;

import multivariate_timeseriesweka.measures.EuclideanDistance_I;
import static utilities.InstanceTools.findMinDistance;
import utilities.generic_storage.Pair;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Aaron
 */
public class NN_ED_I extends AbstractClassifier{
    
    Instances train;
    EuclideanDistance_I I;
    public NN_ED_I(){
        I = new EuclideanDistance_I();
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        train = data;
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception{
        Pair<Instance, Double> minD = findMinDistance(train, instance, I);
        return minD.var1.classValue();
    }
}
