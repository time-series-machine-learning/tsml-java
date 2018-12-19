/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package multivariate_timeseriesweka;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Supplier;
import multivariate_timeseriesweka.classifiers.ConcatenateClassifier;
import multivariate_timeseriesweka.classifiers.NN_DTW_A;
import multivariate_timeseriesweka.classifiers.NN_DTW_D;
import multivariate_timeseriesweka.classifiers.NN_DTW_I;
import multivariate_timeseriesweka.classifiers.NN_ED_D;
import multivariate_timeseriesweka.classifiers.NN_ED_I;
import multivariate_timeseriesweka.ensembles.IndependentDimensionEnsemble;
import timeseriesweka.classifiers.ShapeletTransformClassifier;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.DTW1NN;
import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.RandomForest;

/**
 *
 * @author raj09hxu
 */
public class DefaultClassifiers {
    
        
    public static final Map<String, Supplier<Classifier>> CLASSIFIERS;
    static {
        Map<String, Supplier<Classifier>> map = new HashMap();
        map.put("DTW_A", DefaultClassifiers::createDTW_A);
        map.put("DTW_D", DefaultClassifiers::createDTW_D);
        map.put("DTW_I", DefaultClassifiers::createDTW_I);
        map.put("ED_I", DefaultClassifiers::createED_D);
        map.put("ED_D", DefaultClassifiers::createED_I);
        map.put("RotationForest", DefaultClassifiers::createRotationForest);
        map.put("RandomForest", DefaultClassifiers::createRandomForest);
        map.put("1NN_DTW",DefaultClassifiers::create1NNDTW);
        map.put("MLP",DefaultClassifiers::createMultilayerPerceptron);
        map.put("SMO",DefaultClassifiers::createSMO);
        map.put("RotationForest_concat", DefaultClassifiers::createRotationForest_concat);
        map.put("RandomForest_concat", DefaultClassifiers::createRandomForest_concat);
        map.put("1NN_DTW_concat",DefaultClassifiers::create1NNDTW_concat);
        map.put("MLP_concat",DefaultClassifiers::createMultilayerPerceptron_concat);
        map.put("SMO_concat",DefaultClassifiers::createSMO_concat);
        map.put("ST_concat",DefaultClassifiers::createST_concat);
        CLASSIFIERS = Collections.unmodifiableMap(map);
    }
    
    public static Classifier createDTW_A(){
        NN_DTW_A A = new NN_DTW_A();
        A.setR(0.2); //20%
        return A;
    }
    
    public static Classifier createDTW_I(){
       NN_DTW_I nn = new NN_DTW_I();
       nn.setR(0.2);
       return nn;
    }
    
    public static Classifier createDTW_D(){
       NN_DTW_D nn = new NN_DTW_D();
       nn.setR(0.2);
       return nn;
    }
    
    public static Classifier createED_I(){
       return new NN_ED_I();
    }
    
    public static Classifier createED_D(){
       return new NN_ED_D();
    }
    
    
    public static Classifier createRotationForest(){
        RotationForest rf = new RotationForest();
        rf.setNumIterations(50);
        
        Classifier c = new IndependentDimensionEnsemble(rf);
        return c;
    }
    
    public static Classifier createRandomForest(){
        RandomForest rf = new RandomForest();
        rf.setNumTrees(500);
        
        Classifier c = new IndependentDimensionEnsemble(rf);
        return c;
    }
    
    
    public static Classifier create1NNDTW(){
        DTW1NN nn = new DTW1NN();        
        Classifier c = new IndependentDimensionEnsemble(nn);
        return c;
    }
    
    public static Classifier createMultilayerPerceptron(){
        MultilayerPerceptron mlp = new MultilayerPerceptron();     
        Classifier c = new IndependentDimensionEnsemble(mlp);
        return c;
    }
    
    public static Classifier createSMO(){
        SMO svmq =new SMO();
//Assumes no missing, all real valued and a discrete class variable        
        svmq.turnChecksOff();
        PolyKernel kq = new PolyKernel();
        kq.setExponent(2);
        svmq.setKernel(kq);
        Classifier c = new IndependentDimensionEnsemble(svmq);
        return c;
    }
    
        public static Classifier createRotationForest_concat(){
        RotationForest rf = new RotationForest();
        rf.setNumIterations(50);
        
        Classifier c = new ConcatenateClassifier(rf);
        return c;
    }
    
    public static Classifier createRandomForest_concat(){
        RandomForest rf = new RandomForest();
        rf.setNumTrees(500);
        
        Classifier c = new ConcatenateClassifier(rf);
        return c;
    }
    
    
    public static Classifier create1NNDTW_concat(){
        DTW1NN nn = new DTW1NN();        
        Classifier c = new ConcatenateClassifier(nn);
        return c;
    }
    
    public static Classifier createMultilayerPerceptron_concat(){
        MultilayerPerceptron mlp = new MultilayerPerceptron();     
        Classifier c = new ConcatenateClassifier(mlp);
        return c;
    }
    
    public static Classifier createSMO_concat(){
        SMO svmq =new SMO();
//Assumes no missing, all real valued and a discrete class variable        
        svmq.turnChecksOff();
        PolyKernel kq = new PolyKernel();
        kq.setExponent(2);
        svmq.setKernel(kq);
        Classifier c = new ConcatenateClassifier(svmq);
        return c;
    }
    
    
    public static Classifier createST_concat(){
        return new ConcatenateClassifier(new ShapeletTransformClassifier());
    }
    
    
}
